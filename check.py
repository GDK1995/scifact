# файл где я просто беру запросы
import ir_datasets
import config

# Load datasets
query_dataset = ir_datasets.load(config.PATH_TEST)
query_docs = list(query_dataset.queries_iter())

print(query_docs[66].text)