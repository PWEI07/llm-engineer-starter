I found the documentation of Document AI is really confusing, therefore I resort to Unstructured.io to use its OCR 
functionality. 

# mechanism
This project extract the layout-aware text representation of the original pdf document, perserving the spatial 
relanvancy relationship. it further embedding the extracted information and store the resulting vectors into a 
light-weighted vector database, and 
build a RAG pipeline including embedding redundent filter and reranker to improve the retrieval result when faced 
with potential redundent information. It can answer medical related questions in a fast and reliable way, it will also 
highlight the relavent text based on 
user's question in red 
bounding boxes in a generated PDF to facilitate review.

The project uses unstructured.io (todo introduce this package. its opensoure version and API version, what's its 
advantage)
The main program first uses OCR model in Unstructured to extract the content from the pdf, from which i also get 
bounding boxes 
around each 
text piece so i can use them to restore the layout-aware text representation of the original document. I 
It then created a RAG 
system using the BGE-large-en embedding (todo insert this hyperlink https://huggingface.co/BAAI/bge-large-en also 
introduce this embedding, what's its innovation/superiority) 

# usage
## configuration
The demo (todo insert this link into word 'demo' https://huggingface.co/BAAI/bge-large-en) is run on a Macbook with M1 chip.

To get the layout-aware representation and build a RAG-based Q&A system, run the main() function in submission.py, you 
need to provide the path_to_case_pdf parameter to specify the location of the PDF 
containing patient information to be analyzed.

There's additional parameters you can specify, the most 2 important ones are
llm_path: the path to the GGUF quantized LLM model. (TODO talk about what's GGUF and why does it help to reduce 
memory usage and speed up inference). Here I use mistral-7B (todo introduce mistral-7b and what's its innovation and 
superior performance)

# future research
Time permitted, we can try (todo suggest ways to better extract information from pdf, especially ways to use GCP 
document AI services). the unstructured package worked reliabily in most places in the sample PDF, however in places 
where there's black bar anomonizing patient personal information, it sometimes misread. 

Also, we should find ways to associate date with each event. In GCP document AI services, we can use specify parent 
and 
child fields and whether certain field must appear. for example, we can specify a medical event must be accompanied by 
a date. By ensuring there's the corresponding date for each medical event we retrieved, we can use the date as 
metadata when storing them into the vector database and prompt the LLM to prioritize the latest information when 
answering question (in medical records, the patient condition change over time, so a same for example test result 
will show different evaluation result through time and when answering a question, we should resort to the latest 
information as facts)

we can also fine-tune foundation OCR model to achieve higher extraction accuracy. i.e. we can first manually label some 
document, and use them as data to "teach" the pre-trained model to better adapt to our desiered extraction outcome. 

We also can take into account the element type identified in OCR, so that we better know what information are 
medical event related and what are not for example, in the example pdf, the "4/2/2024 1:22:35 
PM EDT..." appeared multiple times in the document header but it's more likely an indication of the time the 
document is faxed, instead of about the exact time a medical event occured. 