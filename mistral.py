import PyPDF2
from pprint import pprint
from getpass import getpass
from haystack import Document
from haystack import Pipeline
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor,PromptModel, PromptTemplate, PromptNode
HF_TOKEN = getpass("Your Hugging Face Token")
pdf_file_path = "/content/abc.pdf"  #Replace with the path to your PDF file

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

pdf_text = extract_text_from_pdf(pdf_file_path)


doc = Document(
    content=pdf_text,
    meta={"pdf_path": pdf_file_path}
)

docs = [doc]

processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
    split_overlap=0,
)
preprocessed_docs = processor.process(docs)

document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(preprocessed_docs)
retriever = BM25Retriever(document_store, top_k=2)

qa_template = PromptTemplate(prompt=
  """ Using exclusively the information contained in the context, answer only the question asked without adding
  suggestions for possible questions, and respond exclusively in English. If the answer cannot be deduced from the
  context, 
  respond: "Not sure because not relevant to the context.
  Context: {join(documents)};
  Question: {query}
  """)

prompt_node = PromptNode(
    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=HF_TOKEN,
    default_prompt_template=qa_template,
    max_length=500,
    model_kwargs={"model_max_length": 5000}
)

rag_pipeline = Pipeline()
rag_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
rag_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

print_answer = lambda out: pprint(out["results"][0].strip())

question_list = ["invoice no in the document",'Seller address in the document?', 'Seller Tax Id in the document?',
                 'Client Tax Id in the document?', 'IBAN no in the document?']
for question in question_list:
    print_answer(rag_pipeline.run(query=question))