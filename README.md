# Amazon Digital Music Reviews
We use an Amazon digital music review dataset to: 
1. Trial several models that aim to predict the score (1-5*) from the review text. 
2. Explore the text to look for useful business insights. 

# IMPORTANT
Dataset: Amazon Reviews - Digital Music <br />
Source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ <br />
The dataset is not included in this Git repository at present, as even when compressed it is too large. You will have to download it yourself from the link above if you want to rerun any of this analysis. <br />
I am looking into Git LFS to get around this restriction and will hopefully resolve this shortly. 

# Structure & Guidance
The four numbered scripts (1 - 4) in the top level contain the full project and include narrative commentary. <br />

"/gcp notebooks" contains copies of scripts 1 and 3 in Jupyter Notebook form, with some small additions to allow me to run them on Google Cloud Platform's Vertex AI Workbench. This was due to long run times on my local machine. These notebooks also have full commentary, incuding any additional code not contained in the main scripts.  <br />

There are two folders not currently included in this GitHub repo: "/raw data" that contains the data as-downloaded from the link above, and "/processed data" that holds the data after cleaning and processing. These are not included due to file size constraints as discussed previously, but the provided link and script 1 are sufficient to recreate the missing files.
