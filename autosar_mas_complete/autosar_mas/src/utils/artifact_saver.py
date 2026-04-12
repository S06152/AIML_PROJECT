"""
artifact_saver.py — Organized artifact storage for AUTOSAR MAS project.

Saves all agent-generated artifacts to a structured folder hierarchy:
    artifacts/
    └── <session_id>/
        ├── 01_requirements/    ← ProductManagerAgent output
        ├── 02_architecture/    ← ArchitectAgent output
        ├── 03_source_code/     ← DeveloperAgent output
        ├── 04_test_cases/      ← QAAgent output
        └── 05_reviews/         ← CodeReviewAgent output

Design Pattern: Utility class (stateless static methods + class methods).
MNC Standard: Follows separation of concerns, single responsibility principle.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import logger
from src.utils.exception import CustomException


# ---------------------------------------------------------------------------
# Artifact Folder Map: state key → (subfolder_name, file_extension, label)
# ---------------------------------------------------------------------------
_ARTIFACT_MAP: Dict[str, tuple] = {
    "product_spec":  ("01_requirements",  "md",  "Requirements"),
    "architecture":  ("02_architecture",  "md",  "Architecture"),
    "code":          ("03_source_code",   "md",  "Source_Code"),
    "tests":         ("04_test_cases",    "md",  "Test_Cases"),
    "review":        ("05_reviews",       "md",  "Review_Report"),
}

# Root output directory (relative to project root)
_ARTIFACTS_ROOT: str = "artifacts"


class ArtifactSaver:
    """
    Handles creation and saving of all AUTOSAR MAS agent artifacts.

    Responsibilities:
        - Create a timestamped session folder for each workflow run.
        - Save each agent's output to the appropriate subfolder.
        - Return a manifest dict of saved file paths for UI display.

    Design:
        - Session-based: each workflow run gets a unique folder.
        - Non-destructive: existing artifacts are never overwritten.
        - Fail-safe: individual save failures are logged but do not abort the workflow.

    Usage:
        saver = ArtifactSaver(session_id="2026-01-01_12-00-00")
        paths = saver.save_all(final_state)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Initialize ArtifactSaver with a session-specific output directory.

        Args:
            session_id (Optional[str]): Custom session identifier.
                                        Defaults to current timestamp if None.
        """
        try:
            if session_id is None:
                session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self._session_id: str = session_id
            self._session_dir: Path = Path(_ARTIFACTS_ROOT) / session_id

            logger.info("ArtifactSaver initialized. Session dir: %s", self._session_dir)

        except Exception as e:
            raise CustomException(e, sys) from e

    def _ensure_dir(self, folder_name: str) -> Path:
        """
        Create the artifact subfolder if it does not already exist.

        Args:
            folder_name (str): Subfolder name under the session directory.

        Returns:
            Path: Absolute path to the created/existing subfolder.
        """
        target: Path = self._session_dir / folder_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_artifact(
        self,
        artifact_key: str,
        content: str,
        module_name: str = "MODULE",
    ) -> Optional[str]:
        """
        Save a single agent artifact to the appropriate subfolder.

        Args:
            artifact_key (str): State key name (e.g., 'product_spec').
            content      (str): Artifact content to write.
            module_name  (str): AUTOSAR module name for the filename.

        Returns:
            Optional[str]: Absolute path to the saved file, or None on failure.
        """
        if artifact_key not in _ARTIFACT_MAP:
            logger.warning("Unknown artifact key: '%s'. Skipping save.", artifact_key)
            return None

        if not content or not content.strip():
            logger.warning("Empty content for artifact '%s'. Skipping.", artifact_key)
            return None

        try:
            folder_name, extension, label = _ARTIFACT_MAP[artifact_key]
            target_dir: Path = self._ensure_dir(folder_name)

            # Sanitize module name for filesystem safety
            safe_module = module_name.replace(" ", "_").replace("/", "-")
            filename: str = f"{safe_module}_{label}.{extension}"
            file_path: Path = target_dir / filename

            # Add metadata header to every artifact
            header = (
                f"# AUTOSAR MAS — {label}\n"
                f"**Module**: {module_name}  \n"
                f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
                f"**Session**: {self._session_id}  \n\n"
                "---\n\n"
            )

            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(header + content)

            logger.info("Artifact saved: %s", file_path)
            return str(file_path)

        except Exception as e:
            logger.error("Failed to save artifact '%s': %s", artifact_key, str(e))
            return None

    def save_all(
        self,
        final_state: dict,
        module_name: str = "MODULE",
    ) -> Dict[str, str]:
        """
        Save all agent artifacts from the final workflow state.

        Args:
            final_state (dict): DevTeamState containing all agent outputs.
            module_name (str) : AUTOSAR module name for filenames.

        Returns:
            Dict[str, str]: Mapping of artifact_key → saved file path.
                            Keys with failed saves are excluded.
        """
        saved_paths: Dict[str, str] = {}

        for key in _ARTIFACT_MAP:
            content = final_state.get(key, "")
            if content:
                path = self.save_artifact(key, content, module_name)
                if path:
                    saved_paths[key] = path

        logger.info(
            "ArtifactSaver.save_all() complete. Saved %d / %d artifacts.",
            len(saved_paths),
            len(_ARTIFACT_MAP),
        )
        return saved_paths

    @property
    def session_dir(self) -> str:
        """Return the session output directory path as a string."""
        return str(self._session_dir)
