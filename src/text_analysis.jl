# text analysis for irp
using PDFIO, TextAnalysis, Transformers

# load data
loc = joinpath("data", "singapore", "Singapore_National_Artificial_Intelligence_Strategy_2019.pdf")
tgt = joinpath("data", "singapore", "extract.txt")

"""
​```
    getPDFText(src, out) -> Dict 
​```
- src - Input PDF file path from where text is to be extracted
- out - Output TXT file path where the output will be written
return - A dictionary containing metadata of the document
"""
function getPDFText(src, out)
    # handle that can be used for subsequence operations on the document.
    doc = pdDocOpen(src)
    # Metadata extracted from the PDF document. 
    # This value is retained and returned as the return from the function. 
    docinfo = pdDocGetInfo(doc) 
    open(out, "w") do io
        # Returns number of pages in the document       
        npage = pdDocGetPageCount(doc)
        for i=1:npage
            # handle to the specific page given the number index. 
            page = pdDocGetPage(doc, i)
            # Extract text from the page and write it to the output file.
            pdPageExtractText(io, page)
        end
    end
    # Close the document handle - the doc handle should not be used after this call
    pdDocClose(doc)
    return docinfo
end

# read text from pdf
getPDFText(loc, tgt)
sd = StringDocument(read(tgt, String))

# clean text
remove_case!(sd)
text(sd)

# sentiment analysis
using PythonCall, CondaPkg
CondaPkg.add("transformers")
CondaPkg.add("torch")
tf = pyimport("transformers")
classifier = pipeline("sentiment-analysis")