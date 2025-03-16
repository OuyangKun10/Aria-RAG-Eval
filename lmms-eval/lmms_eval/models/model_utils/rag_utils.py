import os
import json
import random


def load_rag_file(file):
    rag = {}
    if file is not None:
        with open(file, "r") as f:
            for line in f:
                item = json.loads(line)
                vid = os.path.basename(item['video_path'])
                rag[vid] = []
                sim2fid = {}
                for img_path, img_score in zip(item['rag_images'], item['rag_sim']):
                    frame_sec = int(os.path.splitext(os.path.basename(img_path))[0]) / item['fps']
                    sim2fid[img_score] = sim2fid.get(img_score, []) + [frame_sec]
                for img_score, frame_secs in sim2fid.items():
                    random.shuffle(frame_secs)
                sorted_scores = sorted(sim2fid.keys(), reverse=True)
                for score in sorted_scores:
                    rag[vid] += sim2fid[score]
    print("===", len(rag))
    return rag


