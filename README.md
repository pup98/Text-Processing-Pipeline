# Text-Processing-Pipeline (NLP)
### Worflow
* First, save the links to be scraped in a csv with random URL'id (here input.csv).
* Then scrape (I used BeautifulSoup) the links and save the output text files to a output directory (here output_data).
* The NLP processing script takes in a input directory (output) where text files are saved and a csv of URL_ID's (can be arbitary numbers) and URL's.
* Output: Script dumps the output into a csv with all the calculated fields and their corresponding URL_ID and URL as a row.

Example Output: <br />
| URL_ID  | URL  | Avg Sentence Length | Count of Complex Words | Fog Index |
| -------:|-----:|:-------------------:| ----------------------:| ---------:|
| -------:|-----:|:-------------------:| ----------------------:| ---------:|
| -------:|-----:|:-------------------:| ----------------------:| ---------:|
| -------:|-----:|:-------------------:| ----------------------:| ---------:|


### Note
The URL's I have chosen have a similar html structure .. so I was able to scrape 100 of websites! <br />
You can modify the scirpt as per your need. 
