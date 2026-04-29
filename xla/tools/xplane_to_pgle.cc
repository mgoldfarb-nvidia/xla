/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// xplane_to_pgle: standalone CLI wrapper around
// xla::ConvertXplaneToProfiledInstructionsProto /
// xla::ConvertXplaneUnderLogdirToProfiledInstructionsProto. Reads an
// xplane.pb (or scans a tensorboard logdir for one) and writes a
// tensorflow::profiler::ProfiledInstructionsProto suitable for
// --xla_gpu_pgle_profile_file_or_directory_path. Same C++ entry
// points as jaxlib's `get_fdo_profile` Python binding, but doesn't
// require a JAX runtime.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/python/xplane_to_profile_instructions.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {
const char* const kUsage = R"(
xplane_to_pgle — convert an xplane.pb (or a tensorboard logdir
containing one) into a ProfiledInstructionsProto suitable for the
LHS PGLE estimator.

Required (exactly one of):
  --xspace=PATH    Single xplane.pb file.
  --logdir=PATH    Tensorboard run directory; the tool scans for the
                   most recent xplane.pb under it (delegates to
                   ConvertXplaneUnderLogdirToProfiledInstructionsProto).

Required:
  --output=PATH    Where to write the ProfiledInstructionsProto.
                   Format inferred from extension:
                     .pbtxt  text (default)
                     .pb     binary

Examples:
  xplane_to_pgle \
      --logdir=outputs/lyris/lyris-hints-zero-serial-v3-stable_*/maxtext/.../tensorboard/plugins/profile/2026_04_29_06_57_29 \
      --output=/tmp/v3stable.pbtxt

  xplane_to_pgle \
      --xspace=foo.xplane.pb --output=/tmp/foo.pb
)";
}  // namespace

namespace xla {

namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  if (s.size() < suffix.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

absl::Status WriteProto(const tensorflow::profiler::ProfiledInstructionsProto&
                            proto,
                        const std::string& path) {
  if (EndsWith(path, ".pbtxt")) {
    std::string text;
    if (!tsl::protobuf::TextFormat::PrintToString(proto, &text)) {
      return absl::InternalError("failed to serialize as text proto");
    }
    std::ofstream os(path);
    os << text;
    std::cerr << "Wrote " << text.size() << " chars (text) -> " << path
              << "\n";
    return absl::OkStatus();
  }
  // Binary fallback (matches `.pb` and any unknown extension).
  std::string serialized;
  if (!proto.SerializeToString(&serialized)) {
    return absl::InternalError("failed to serialize as binary proto");
  }
  std::ofstream os(path, std::ios::binary);
  os.write(serialized.data(), serialized.size());
  std::cerr << "Wrote " << serialized.size() << " bytes (binary) -> "
            << path << "\n";
  return absl::OkStatus();
}

absl::Status RunFromXspace(const std::string& xspace_path,
                           const std::string& output_path) {
  tensorflow::profiler::XSpace xspace;
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(), xspace_path,
                                          &xspace));
  std::vector<tensorflow::profiler::XSpace> xspaces;
  xspaces.push_back(std::move(xspace));
  tensorflow::profiler::ProfiledInstructionsProto profile;
  TF_RETURN_IF_ERROR(
      ConvertXplaneToProfiledInstructionsProto(std::move(xspaces), &profile));
  std::cerr << "Aggregated " << profile.costs_size() << " cost entries, "
            << profile.latencies_size() << " latency entries from "
            << xspace_path << "\n";
  return WriteProto(profile, output_path);
}

absl::Status RunFromLogdir(const std::string& logdir,
                           const std::string& output_path) {
  tensorflow::profiler::ProfiledInstructionsProto profile;
  TF_RETURN_IF_ERROR(
      ConvertXplaneUnderLogdirToProfiledInstructionsProto(logdir, &profile));
  std::cerr << "Aggregated " << profile.costs_size() << " cost entries, "
            << profile.latencies_size() << " latency entries from "
            << logdir << "\n";
  return WriteProto(profile, output_path);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::string xspace;
  std::string logdir;
  std::string output;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("xspace", &xspace, "Path to a single .xplane.pb file"),
      tsl::Flag("logdir", &logdir,
                "Tensorboard run directory containing one or more "
                ".xplane.pb files"),
      tsl::Flag("output", &output,
                "Output path. Extension determines format: .pbtxt = text, "
                ".pb (or other) = binary"),
  };

  std::string usage(kUsage);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (!parse_ok || output.empty() || (xspace.empty() == logdir.empty())) {
    std::cerr << usage;
    std::cerr << "\nFlags:\n" << tsl::Flags::Usage(argv[0], flag_list);
    return 1;
  }

  absl::Status s = xspace.empty() ? xla::RunFromLogdir(logdir, output)
                                  : xla::RunFromXspace(xspace, output);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s << "\n";
    return 2;
  }
  return 0;
}
