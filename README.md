This repository contains the necesary code and data to reproduce the study presented in:

R.A. Cardenas, K.S. Bello. "Labor market demand analysis for engineering majors in Peru using Shallow Parsing and Topic Modeling". In Poster Session of the Machine Learning Summer School Kyoto 2015, Kyoto, Japan.

The dataset used for this study consisted of more than 200000 job ads extracted from several job hunting websites in Peru. Data for other Latin American countries is available as well, although not included in the analysis.

Each dataset used in the models used is available here or upon request, and explained below.
<ul>
	<li><b>Tokenized job ads:</b> more than 900k job ads extracted from Latin American websites. The NLTK tokenizer was extended to capture technical words typical of these kind of advertisement (check the preprocessing folder). <b>[available upon request]</b></li>
	<li><b>Shallow Parsing models [annotated data.zip]</b>: Consisting of 800 job ads, each one tokenized and manually annotated with POS tag information (<a href="http://nlp.lsi.upc.edu/freeling/doc/tagsets/tagset-es.html">EAGLE</a> format for Spanish data) and Entity Label in BIO format</li>
	<li><b>Topic models</b>: Consisting of nearly 9000 job ads sampled from the database, tokenized and filtered from low-frequency words and tokens of no interest (phone numbers, salary, office hours, emails, urls). Then, the shallow parsers extract the relevant phrases.
		<ul>
			<li>[filtered_FULL_TEXT_data.zip]: Tokenized and filtered complete ads.</li>
			<li>[filtered_CHUNKS_TEXT_data.zip]: Text extracted by the shallow parsers from tokenized and filtered complete ads.</li>
		</ul>
	</li>
</ul>