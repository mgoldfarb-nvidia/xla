/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

// xplane_summary: print a per-plane summary of an xplane.pb so we can
// see how much real GPU wall time is covered by kernel events vs idle
// gaps. Used to calibrate predict_overlap.py against measured step
// time (predicted ~1.4s vs measured ~2.5s on module_43775; this tool
// answers "where's the missing 1.1s — kernel slowdown or GPU idle?").

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {
const char* const kUsage = R"(
xplane_summary — dump per-plane / per-line totals from an xplane.pb so
the gap between predicted and measured step time can be attributed to
GPU idle vs kernel-slowdown vs host-side overhead.

Required:
  --xspace=PATH    Path to the xplane.pb file.

Output (per plane):
  * line count, event count
  * wall span    = max(event_end) - min(event_start)
  * busy span    = sum(event_durations)
  * idle         = wall - busy
  * busy fraction
  * top 10 events by total time (sums across all instances of same name)
)";
}

int main(int argc, char** argv) {
  std::string xspace_path;
  std::vector<tsl::Flag> flags = {
      tsl::Flag("xspace", &xspace_path, "Path to .xplane.pb"),
  };
  if (!tsl::Flags::Parse(&argc, argv, flags) || xspace_path.empty()) {
    std::cerr << kUsage << "\n" << tsl::Flags::Usage(argv[0], flags);
    return 1;
  }
  tsl::port::InitMain(argv[0], &argc, &argv);

  tensorflow::profiler::XSpace xspace;
  TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), xspace_path, &xspace));

  std::cerr << "Loaded XSpace: " << xspace.planes_size() << " plane(s)\n\n";

  for (const auto& plane : xspace.planes()) {
    // Build event_metadata id → name lookup so we can resolve events.
    std::map<int64_t, std::string> em;
    for (const auto& kv : plane.event_metadata()) {
      em[kv.first] = kv.second.name();
    }

    int64_t event_count = 0;
    int64_t min_start = INT64_MAX;
    int64_t max_end = 0;
    int64_t busy = 0;
    std::map<std::string, int64_t> per_name_total_ps;
    std::map<std::string, int64_t> per_name_count;
    for (const auto& line : plane.lines()) {
      for (const auto& event : line.events()) {
        event_count++;
        int64_t s = event.offset_ps();
        int64_t d = event.duration_ps();
        int64_t e = s + d;
        if (s < min_start) min_start = s;
        if (e > max_end) max_end = e;
        busy += d;
        const auto it = em.find(event.metadata_id());
        std::string name = (it != em.end()) ? it->second : "<unknown>";
        per_name_total_ps[name] += d;
        per_name_count[name]++;
      }
    }
    if (event_count == 0) continue;
    int64_t wall_ps = max_end - min_start;
    int64_t idle_ps = wall_ps - busy;

    std::cout << "=== " << plane.name() << " ===\n";
    std::cout << "  lines:   " << plane.lines_size() << "\n";
    std::cout << "  events:  " << event_count << "\n";
    std::cout << "  wall:    " << wall_ps / 1e9 << " ms"
              << "  (min_start=" << min_start / 1e9
              << " ms, max_end=" << max_end / 1e9 << " ms)\n";
    std::cout << "  busy:    " << busy / 1e9 << " ms"
              << "  (sum of all event durations)\n";
    std::cout << "  idle:    " << idle_ps / 1e9 << " ms"
              << "  (wall - busy)\n";
    if (wall_ps > 0) {
      std::cout << "  busy_pct: "
                << 100.0 * busy / static_cast<double>(wall_ps) << "%\n";
    }
    // Top 10 names by total time.
    std::vector<std::pair<std::string, int64_t>> sorted;
    sorted.reserve(per_name_total_ps.size());
    for (const auto& kv : per_name_total_ps) sorted.push_back(kv);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                return a.second > b.second;
              });
    int n = std::min<int>(10, sorted.size());
    std::cout << "  top " << n << " events by total time:\n";
    for (int i = 0; i < n; ++i) {
      std::cout << "    " << sorted[i].second / 1e9 << " ms  "
                << "x" << per_name_count[sorted[i].first] << "  "
                << sorted[i].first << "\n";
    }
    std::cout << "\n";
  }
  return 0;
}
