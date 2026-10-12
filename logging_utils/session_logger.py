import json
from datetime import datetime
from pathlib import Path


class SessionLogger:

    def __init__(self, session_id, episode_number=1):

        self.session_id = session_id
        self.episode_number = episode_number

        self.agent_dir = Path("logs/agent")
        self.conversation_dir = Path("logs/conversations")
        self.experiment_dir = Path("logs/experiments")
        
        self.agent_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.conversation_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.agent_file = (
            self.agent_dir /
            f"{session_id}.jsonl"
        )

        self.conversation_file = (
            self.conversation_dir /
            f"{session_id}.jsonl"
        )

        self.experiment_file = (
            self.experiment_dir /
            f"{session_id}.jsonl"
        )

    def log_agent(
        self,
        event,
        data=None,
    ):

        record = {
            "timestamp":
                datetime.now().isoformat(),

            "event":
                event,

            "data":
                data or {},
        }

        with open(
            self.agent_file,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def log_message(
        self,
        role,
        content,
    ):

        record = {
            "timestamp":
                datetime.now().isoformat(),

            "role":
                role,

            "content":
                content,
        }

        with open(
            self.conversation_file,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def log_experiment(
        self,
        experiment_name,
        data=None,
    ):

        record = {
            "timestamp":
                datetime.now().isoformat(),

            "experiment_name":
                experiment_name,

            "data":
                data or {},
        }

        with open(
            self.experiment_file,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )