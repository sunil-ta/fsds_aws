import subprocess

# from ../src import ingest_data, score, train

subprocess.run(["python", "-m", "scripts.ingest_data"])
subprocess.run(["python", "-m", "scripts.train"])
subprocess.run(["python", "-m", "scripts.score"])
