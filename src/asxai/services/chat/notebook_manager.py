import os
import json
import shutil
import time
import urllib
from urllib.error import HTTPError


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

        await self.search_cleanup(task_id)

        return len(history) - len(new_history)

    async def set_paper_score(self, task_id, paperIds, scores: dict):
        # updating scores in chat
        chat_path = self.get_chat_path(task_id)
        print(scores)
        scores = {k: val for k, val in scores.items() if 'score' in k}
        print(scores)
        if not paperIds:
            return []

        if not isinstance(paperIds, list):
            paperIds = [paperIds]

        if os.path.isfile(chat_path):
            with open(chat_path, "r") as f:
                chat_history = json.load(f)

            for msg in chat_history:
                updated_papers = []
                original_papers = msg.get("papers", [])
                for paper in original_papers or []:
                    if paper.get('paperId', '') in paperIds:
                        paper = {**paper, **scores}
                        print("paper found in chat")
                    updated_papers.append(paper)

                if updated_papers:
                    msg['papers'] = updated_papers

            with open(chat_path, "w") as f:
                json.dump(chat_history, f, indent=2)

        # updating scores in search results
        search_path = self.get_search_path(task_id)
        if os.path.isfile(search_path):
            with open(search_path, "r") as f:
                search_history = json.load(f)

            for paper in search_history:
                if paper.get('paperId', '') in paperIds:
                    paper = {**paper, **scores}
                    print("paper found in search")

            with open(search_path, "w") as f:
                json.dump(search_history, f, indent=2)

        return paperIds

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

        await self.search_cleanup(task_id)

        return len(history) - len(new_history)

    async def build_formatted_citation(self, doi_url, style='nature', get_json=False, verbose=True):
        BASE_URL = 'https://doi.org/'
        bibtex = None

        if len(doi_url) == 0:
            return None

        if not doi_url.startswith('http'):
            doi_url = BASE_URL + doi_url

        if BASE_URL not in doi_url:
            raise ValueError("Incorrect doi_url: {}! It might starts with {}".format(
                doi_url, BASE_URL))

        req = urllib.request.Request(doi_url)

        if get_json:
            args = "application/citeproc+json"
        else:
            args = "text/bibliography; style={style}".format(style=style)

        req.add_header('Accept', args)

        try:
            with urllib.request.urlopen(req) as f:
                bibtex = f.read().decode('utf-8')
        except HTTPError as e:
            if e.code == 404:
                print('DOI not found: {}'.format(doi_url))
            else:
                print('Service unavailable.')

        if not get_json:
            if bibtex[0].isdigit():
                bibtex = bibtex.split('.', 1)[-1]  # .strip('\n')

        return bibtex

    async def get_citation_list(self, task_id, style: str = 'nature'):
        search_path = self.get_search_path(task_id)
        search_history = []
        if os.path.isfile(search_path):
            with open(search_path, "r") as f:
                search_history = json.load(f)
            doi_list = [pl.get('doi', None)
                        for pl in search_history if pl.get('doi', None)]

        if not doi_list:
            return ["No valid DOI"]

        refList = []
        for doi in doi_list:
            refList.append(await self.build_formatted_citation(
                doi, style=style, get_json=False, verbose=False))

        return refList, doi_list

    async def set_update_time(self, task_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        update_timestamp = time.time()
        with open(chat_path, "r") as f:
            chat_history = json.load(f)

        for m in chat_history:
            m['timestamp_update'] = update_timestamp

        with open(chat_path, "w") as f:
            json.dump(chat_history, f)

        return update_timestamp

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
                fpath = os.path.join(searchQ_path, fname)
                with open(fpath, "r") as f:
                    query_results.extend(json.load(f))
                os.remove(fpath)
            os.rmdir(searchQ_path)
        return query_results

    async def search_cleanup(self, task_id):
        chat_path = self.get_chat_path(task_id)
        if not os.path.isfile(chat_path):
            return 0

        with open(chat_path, "r") as f:
            chat_history = json.load(f)

        all_query_ids = set([m.get("query_id") for m in chat_history if m.get(
            "role") == "assistant"])
        all_paper_ids = set([
            paper.get("paperId") for m in chat_history
            for paper in m.get("papers", []) or []
        ])

        search_history = []
        search_path = self.get_search_path(task_id)
        query_results = self.collect_searches(task_id)
        if os.path.isfile(search_path):
            with open(search_path, "r") as f:
                search_history = json.load(f)
        search_history += query_results

        new_search_history = [
            pl for pl in search_history if pl.get(
                "query_id", '').split('_', 1)[-1] in all_query_ids and pl.get("paperId", '') in all_paper_ids]

        unique_papers = {}
        for paper in new_search_history:
            pid = paper.get("paperId")
            unique_papers[pid] = paper

        new_search_history = list(unique_papers.values())

        with open(search_path, "w") as f:
            json.dump(new_search_history, f)

        print(
            f"DIFF in search history: {len(search_history) - len(new_search_history)} out of {len(search_history)}")
        print(all_paper_ids)
        return len(search_history) - len(new_search_history)

    async def notebooks_cleanup(self, user_id):
        user_dir = os.path.join(self.users_root, user_id)
        if not os.path.exists(user_dir):
            return []

        deleted = []
        kept = []

        for f in os.listdir(user_dir):
            if f.endswith(".chat.json"):
                task_id = f[:-10]
                chat_path = os.path.join(user_dir, f)

                try:
                    with open(chat_path, "r") as j:
                        chat_data = json.load(j)

                    if chat_data:
                        kept.append(task_id)
                        continue

                    deleted.append(task_id)
                    os.remove(chat_path)

                    # Remove related files
                    search_path = os.path.join(user_dir, f"{task_id}.json")
                    summaries_path = os.path.join(
                        user_dir, f"{task_id}.summaries.json")
                    if os.path.isfile(search_path):
                        os.remove(search_path)
                    if os.path.isfile(summaries_path):
                        os.remove(summaries_path)

                    # Remove task_id directory if it exists
                    task_dir = os.path.join(user_dir, task_id)
                    if os.path.isdir(task_dir):
                        shutil.rmtree(task_dir)

                except Exception as e:
                    print(f"Failed to process {f}: {e}")
                    continue

        for f in os.listdir(user_dir):
            if all(not f.startswith(k) for k in kept):
                f_path = os.path.join(user_dir, f)
                try:
                    if os.path.isfile(f_path):
                        os.remove(f_path)
                    elif os.path.isdir(f_path):
                        shutil.rmtree(f_path)
                except Exception as e:
                    print(f"Failed to remove {f_path}: {e}")
                    continue

        return deleted
