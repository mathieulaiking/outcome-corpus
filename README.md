# Primary and Secondary Outcomes Manually Annotated Corpus

## Corpus :

86 COVID-19 Randomized Controlled Trials articles. For each article, a part of the full text or abstract supposed to contain information about Clinical Trial Outcomes have been extracted using Entrez API , a XML parser for the Entrez article format, and then multiple regular expression to detect outcome sections. The goal was to annotate entities and relations useful for the detection of discrepancies between clinical trial registration and publication, which includes : 
* Change in timing of the assessment of the primary outcome
* New primary outcome introduced in the paper
* Registered primary outcome not reported in the paper
* Registered primary endpoint reported as secondary outcome in the paper
* Registered secondary outcome reported as a primary outcome in the paper

We also provide the [annotation guidelines](guidelines.pdf).

## Annotators : 

2 Master students in Epidemiology and Statistics annotated Entities and Relations using [brat](https://brat.nlplab.org/). A third annotator (PhD Student in NLP, with basic knwoledge about clinical trials) did the adjudication. F1 Agreement is available [here](./agreement/f1-instance-agreement.md).
