from __future__ import annotations

import json
from pathlib import Path

from decisionhead.backends import FakeBackend
from decisionhead.bench import (
    bench_consistency,
    bench_latency_by_batch_size,
    bench_stability,
    bench_throughput,
    main,
    render_stability_report_markdown,
    run_bench,
    stability_report_paths,
    write_stability_report,
)


class TestLatencyBench:
    def test_covers_every_requested_batch_size(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_latency_by_batch_size(fb, sizes=(1, 3, 5))
        assert set(result.keys()) == {"1", "3", "5"}

    def test_records_one_entry_per_question(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_latency_by_batch_size(fb, sizes=(1, 3, 5))
        assert len(result["5"]["per_question_s"]) == 5

    def test_separates_first_from_later_questions(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_latency_by_batch_size(fb, sizes=(1, 3, 5))
        assert result["5"]["first_question_s"] is not None
        assert result["5"]["later_questions_mean_s"] is not None

    def test_single_question_has_no_later_mean(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_latency_by_batch_size(fb, sizes=(1,))
        assert result["1"]["later_questions_mean_s"] is None


class TestThroughputBench:
    def test_covers_every_requested_concurrency(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_throughput(fb, concurrencies=(1, 2))
        assert set(result.keys()) == {"1", "2"}

    def test_reports_positive_requests_per_second(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_throughput(fb, concurrencies=(1, 2))
        assert result["2"]["requests_per_s"] > 0

    def test_reports_mean_and_max_latency(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_throughput(fb, concurrencies=(1, 2))
        assert result["2"]["mean_request_latency_s"] is not None
        assert result["2"]["max_request_latency_s"] is not None


class TestConsistencyBench:
    def test_covers_20_fixed_questions(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_consistency(fb, repeats=5)
        assert result["n_questions"] == 20

    def test_deterministic_fake_backend_shows_zero_disagreement(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_consistency(fb, repeats=5)
        assert result["share_any_disagreement"] == 0.0

    def test_per_question_disagreement_count_reported(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_consistency(fb, repeats=5)
        assert len(result["per_question_disagreement_count"]) == 20


class TestStabilityBench:
    def test_covers_20_fixed_questions(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_stability(fb, repeats=5)
        assert result["n_questions"] == 20

    def test_runs_at_concurrency_one(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_stability(fb, repeats=5)
        assert result["concurrency"] == 1

    def test_deterministic_fake_backend_has_exactly_one_distinct_value_per_question(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_stability(fb, repeats=5)
        assert all(c == 1 for c in result["distinct_value_counts"].values())
        assert result["questions_with_multiple_distinct_values"] == 0

    def test_zero_disagreement_against_a_deterministic_backend(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_stability(fb, repeats=5)
        assert result["share_any_disagreement"] == 0.0

    def test_report_is_json_serializable(self) -> None:
        fb = FakeBackend(seed=11)
        result = bench_stability(fb, repeats=5)
        json.dumps(result)  # must not raise

    def test_detects_multiple_distinct_values_from_a_flaky_backend(self) -> None:
        calls = {"n": 0}

        class FlakyBackend:
            model = "flaky"

            def token_distribution(self, prefix_messages, question_text, allowed_tokens):
                calls["n"] += 1
                # strictly increasing with every call, so every repeat of
                # every question gets a distinct probability -- like a
                # batching-endpoint's logprob drift, not like a coin flip
                # that might coincidentally repeat a value.
                p_yes = min(0.05 + 0.001 * calls["n"], 0.95)
                dist = {t: (p_yes if t == "Yes" else 1 - p_yes) for t in allowed_tokens}
                return {**dist, "_unparsed": 0.0, "_raw": {}}

        result = bench_stability(FlakyBackend(), repeats=4)
        noul_questions = [qid for qid in result["distinct_value_counts"] if qid.startswith("noul_")]
        assert any(result["distinct_value_counts"][qid] > 1 for qid in noul_questions)


class TestStabilityReportIo:
    def test_stability_report_paths_from_json_extension(self) -> None:
        assert stability_report_paths("bench_report.json") == (
            "bench_report.stability.json",
            "bench_report.stability.md",
        )

    def test_stability_report_paths_from_extensionless_base(self) -> None:
        assert stability_report_paths("bench_report") == ("bench_report.stability.json", "bench_report.stability.md")

    def test_markdown_report_contains_model_and_every_question(self) -> None:
        fb = FakeBackend(seed=11)
        report = bench_stability(fb, repeats=3)
        md = render_stability_report_markdown(report)
        assert report["model"] in md
        for qid in report["distinct_value_counts"]:
            assert qid in md

    def test_write_stability_report_writes_both_files(self, tmp_path: Path) -> None:
        fb = FakeBackend(seed=11)
        report = bench_stability(fb, repeats=3)
        json_path = tmp_path / "r.stability.json"
        md_path = tmp_path / "r.stability.md"
        write_stability_report(report, str(json_path), str(md_path))
        assert json.loads(json_path.read_text()) == report
        assert "Stability report" in md_path.read_text()

    def test_cli_stability_flag_writes_json_and_markdown(self, tmp_path: Path) -> None:
        out_path = tmp_path / "report.json"
        code = main(["--dry", "--repeats", "3", "--stability", "--out", str(out_path)])
        assert code == 0
        json_path, md_path = stability_report_paths(str(out_path))
        assert Path(json_path).exists()
        assert Path(md_path).exists()

    def test_cli_without_stability_flag_still_writes_the_full_report(self, tmp_path: Path) -> None:
        out_path = tmp_path / "report.json"
        code = main(["--dry", "--repeats", "3", "--out", str(out_path)])
        assert code == 0
        assert out_path.exists()
        report = json.loads(out_path.read_text())
        assert "stability" not in report  # unaffected by the --stability flag's existence


class TestRunBench:
    def test_assembles_all_sections(self) -> None:
        fb = FakeBackend(seed=11)
        report = run_bench(fb, repeats=5)
        assert {"latency_by_batch_size", "throughput_by_concurrency", "consistency", "model"} == set(report.keys())

    def test_report_is_json_serializable(self) -> None:
        fb = FakeBackend(seed=11)
        report = run_bench(fb, repeats=5)
        json.dumps(report)  # must not raise
