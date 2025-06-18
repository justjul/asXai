import os
import json
import shutil
import time


class NotebookManager:
    def __init__(self, users_root):
        self.users_root = users_root

    def get_chat_path(self, task_id):
        return os.path.join(self.users_root, f"{task_id}.chat.json")

    def get_stream_path(self, task_id):
        return os.path.join(self.users_root, f"{task_id}.chat.json")

    def get_search_path(self, task_id):
        return os.path.join(self.users_root, f"{task_id}.json")

    def get_summaries_path(self, task_id):
        return os.path.join(self.users_root, f"{task_id}.summaries.json")

    async def list(self, user_id):
        user_dir = os.path.join(self.users_root, user_id)
        if not os.path.exists(user_dir):
            return []

        notebooks = []
        for f in os.listdir(user_dir):
            if f.endswith(".chat.json"):
                try:
                    with open(os.path.join(user_dir, f), "r") as j:
                        data = json.load(j)
                        notebooks.append({
                            "id": f[:-10],
                            "name": data[-1].get("notebook_id", f[:-10]),
                            "title": data[-1].get("notebook_title", f[:-10]),
                            "date": data[0].get("timestamp", 0)
                        })
                except Exception:
                    continue

        notebooks.sort(key=lambda x: x['date'], reverse=True)
        return notebooks

    async def rename(self, task_id, new_title):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return []

        try:
            with open(chat_path, "r") as f:
                notebook = json.load(f)

            for msg in notebook:
                msg['notebook_title'] = new_title

            with open(chat_path, "w") as f:
                json.dump(notebook, f)
        except Exception:
            notebook = []

        return {"task_id": task_id, "notebook_title": new_title}

    async def delete(self, task_id):
        chat_path = self.get_chat_path(task_id)
        search_path = self.get_search_path(task_id)
        summaries_path = self.get_summaries_path(task_id)
        stream_path = self.get_stream_path(task_id)
        if os.path.isfile(chat_path):
            os.remove(chat_path)
        if os.path.isfile(search_path):
            os.remove(search_path)
        if os.path.isfile(summaries_path):
            os.remove(summaries_path)
        if os.path.isfile(stream_path):
            os.remove(stream_path)
        return {"task_id": task_id, "status": 'deleted'}

    async def load_history(self, task_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return []

        with open(chat_path, "r") as f:
            return json.load(f)

    async def load_query(self, task_id, query_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return []

        with open(chat_path, "r") as f:
            history = json.load(f)

        return [m for m in history if m.get("query_id") == query_id]

    async def delete_query(self, task_id, query_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        with open(chat_path, "r") as f:
            history = json.load(f)

        new_history = [m for m in history if m.get("query_id") != query_id]

        with open(chat_path, "w") as f:
            json.dump(new_history, f)

        self.search_cleanup(task_id)

        return len(history) - len(new_history)

    async def delete_queries_from(self, task_id, query_id, keepUserMsg: bool = False):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        with open(chat_path, "r") as f:
            history = json.load(f)

        role = 'user' if not keepUserMsg else 'assistant'
        query_timestamp = next((
            m["timestamp"] for m in history if m["query_id"] == query_id and m["role"] == role),
            None
        )
        new_history = [
            m for m in history if m.get("timestamp") < query_timestamp
        ]

        with open(chat_path, "w") as f:
            json.dump(new_history, f)

        self.search_cleanup(task_id)

        return len(history) - len(new_history)

    async def chat_cleanup(self, task_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        with open(chat_path, "r") as f:
            chat_history = json.load(f)

        all_query_ids = [m.get("query_id") for m in chat_history if m.get(
            "role") == "assistant"]
        new_chat_history = [m for m in chat_history if m.get(
            "query_id") in all_query_ids]

        with open(chat_path, "w") as f:
            json.dump(new_chat_history, f)

        return len(chat_history) - len(new_chat_history)

    def collect_searches(self, task_id):
        searchQ_path = os.path.join(self.users_root, f"{task_id}")
        query_results = []
        if os.path.isdir(searchQ_path):
            fname_list = os.listdir(searchQ_path)
            for fname in fname_list:
                with open(os.path.join(searchQ_path, fname), "r") as f:
                    query_results.extend(json.load(f))
        return query_results

    async def search_cleanup(self, task_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        with open(chat_path, "r") as f:
            chat_history = json.load(f)

        all_query_ids = set([m.get("query_id") for m in chat_history if m.get(
            "role") == "assistant"])

        search_path = self.get_search_path(task_id)
        query_results = self.collect_searches(task_id)
        if os.path.isfile(search_path):
            with open(search_path, "r") as f:
                search_history = json.load(f)
        else:
            search_history = query_results

        new_search_history = [pl for pl in search_history if pl.get(
            "query_id", '').split('_', 1)[-1] in all_query_ids]

        unique_papers = {}
        for paper in new_search_history:
            pid = paper.get("paperId")
            unique_papers[pid] = paper

        new_search_history = list(unique_papers.values())

        with open(search_path, "w") as f:
            json.dump(new_search_history, f)

        return len(search_history) - len(new_search_history)
