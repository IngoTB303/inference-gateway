"""Tests for experiment shell scripts — check presence, syntax, and structure."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "scripts"
MODAL = REPO_ROOT / "modal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bash_syntax_ok(path: Path) -> bool:
    """Return True if `bash -n <path>` reports no syntax errors."""
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Script presence
# ---------------------------------------------------------------------------


class TestScriptPresence:
    def test_common_sh_exists(self):
        assert (SCRIPTS / "_common.sh").is_file()

    def test_run_experiments_exists(self):
        assert (SCRIPTS / "run_experiments.sh").is_file()

    def test_deploy_modal_vllm_exists(self):
        assert (SCRIPTS / "deploy_modal_vllm.sh").is_file()

    def test_start_stack_exists(self):
        assert (SCRIPTS / "start_stack.sh").is_file()


# ---------------------------------------------------------------------------
# Bash syntax
# ---------------------------------------------------------------------------


class TestBashSyntax:
    def test_common_sh_syntax(self):
        assert bash_syntax_ok(SCRIPTS / "_common.sh"), "_common.sh has syntax errors"

    def test_run_experiments_syntax(self):
        assert bash_syntax_ok(SCRIPTS / "run_experiments.sh"), (
            "run_experiments.sh has syntax errors"
        )

    def test_deploy_modal_vllm_syntax(self):
        assert bash_syntax_ok(SCRIPTS / "deploy_modal_vllm.sh"), (
            "deploy_modal_vllm.sh has syntax errors"
        )

    def test_start_stack_syntax(self):
        assert bash_syntax_ok(SCRIPTS / "start_stack.sh"), (
            "start_stack.sh has syntax errors"
        )


# ---------------------------------------------------------------------------
# Script content — key flags and structure
# ---------------------------------------------------------------------------


class TestRunExperimentsContent:
    def setup_method(self):
        self.src = (SCRIPTS / "run_experiments.sh").read_text()

    def test_sources_common(self):
        assert 'source "$SCRIPT_DIR/_common.sh"' in self.src

    def test_all_three_profiles_present(self):
        for profile in ("standard", "optimized", "hardcore"):
            assert profile in self.src, (
                f"Profile '{profile}' missing from run_experiments.sh"
            )

    def test_all_three_techniques_present(self):
        for technique in ("baseline", "optimized", "hardcore"):
            assert technique in self.src, (
                f"Technique '{technique}' missing from run_experiments.sh"
            )

    def test_wait_for_backend_defined(self):
        assert "wait_for_backend" in self.src

    def test_deploy_flag_handled(self):
        assert "--deploy" in self.src

    def test_crew_py_invoked(self):
        assert "crew.py" in self.src

    def test_crew_vllm_wait_disabled_after_own_wait(self):
        # The script handles its own polling; crew's internal wait should be skipped.
        assert "CREW_VLLM_WAIT_S=0" in self.src


class TestStartStackContent:
    def setup_method(self):
        self.src = (SCRIPTS / "start_stack.sh").read_text()

    def test_starts_two_gateway_instances(self):
        assert "GW1_PID" in self.src
        assert "GW2_PID" in self.src

    def test_uses_uv_run(self):
        assert "uv run python" in self.src

    def test_starts_nginx(self):
        assert "nginx" in self.src

    def test_cleanup_trap_defined(self):
        assert "trap cleanup" in self.src

    def test_different_ports(self):
        assert "PORT1" in self.src
        assert "PORT2" in self.src


class TestDeployModalContent:
    def setup_method(self):
        self.src = (SCRIPTS / "deploy_modal_vllm.sh").read_text()

    def test_standard_profile(self):
        assert "vllm_gemma4.py" in self.src

    def test_optimized_profile(self):
        assert "vllm_gemma4_optimized.py" in self.src

    def test_hardcore_profile(self):
        assert "vllm_gemma4_hardcore.py" in self.src

    def test_hardcore_in_case_statement(self):
        assert "hardcore)" in self.src


# ---------------------------------------------------------------------------
# Modal container files
# ---------------------------------------------------------------------------


class TestModalFiles:
    def test_standard_modal_file_exists(self):
        assert (MODAL / "vllm_gemma4.py").is_file()

    def test_optimized_modal_file_exists(self):
        assert (MODAL / "vllm_gemma4_optimized.py").is_file()

    def test_hardcore_modal_file_exists(self):
        assert (MODAL / "vllm_gemma4_hardcore.py").is_file()

    def test_hardcore_has_fp8_kv_cache(self):
        src = (MODAL / "vllm_gemma4_hardcore.py").read_text()
        assert "--kv-cache-dtype" in src
        assert "fp8" in src

    def test_hardcore_has_chunked_prefill(self):
        src = (MODAL / "vllm_gemma4_hardcore.py").read_text()
        assert "--enable-chunked-prefill" in src

    def test_hardcore_has_prefix_caching(self):
        src = (MODAL / "vllm_gemma4_hardcore.py").read_text()
        assert "--enable-prefix-caching" in src

    def test_hardcore_larger_batch_than_optimized(self):
        hardcore_src = (MODAL / "vllm_gemma4_hardcore.py").read_text()
        optimized_src = (MODAL / "vllm_gemma4_optimized.py").read_text()

        def extract_batch_tokens(src: str) -> int:
            for line in src.splitlines():
                if (
                    "CHUNKED_PREFILL_TOKENS" in line
                    and "=" in line
                    and "#" not in line.split("=")[0]
                ):
                    value = line.split("=")[1].split("#")[0].strip()
                    return int(value)
            return 0

        assert extract_batch_tokens(hardcore_src) > extract_batch_tokens(optimized_src)

    def test_hardcore_more_seqs_than_optimized(self):
        hardcore_src = (MODAL / "vllm_gemma4_hardcore.py").read_text()
        optimized_src = (MODAL / "vllm_gemma4_optimized.py").read_text()

        def extract_max_seqs(src: str) -> int:
            for line in src.splitlines():
                if (
                    "MAX_NUM_SEQS" in line
                    and "=" in line
                    and "#" not in line.split("=")[0]
                ):
                    value = line.split("=")[1].split("#")[0].strip()
                    return int(value)
            return 0

        assert extract_max_seqs(hardcore_src) > extract_max_seqs(optimized_src)

    def test_all_modal_files_serve_gemma4(self):
        for fname in (
            "vllm_gemma4.py",
            "vllm_gemma4_optimized.py",
            "vllm_gemma4_hardcore.py",
        ):
            src = (MODAL / fname).read_text()
            assert "gemma-4-e2b-it" in src, f"{fname} missing gemma-4-e2b-it model name"
