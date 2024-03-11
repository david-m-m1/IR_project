Structure, organization:
In this project, we documented our progress. Each directory is a new version and a step forward in the project based on the tips from the requirements file.
We mainly used colab notebooks for the project there we located the main code we upload in each version.
At the final version, we also added backend and inverted index and search fronted that we submitted for the final engine.
If we made changes to other files besides the notebook, we added them with each new version.

We wrote functionality of each major piece of the engine in few lines at the start of each notebook, also adding here of each step here for understanding the progress:

version1:  creation and search on small data (1000) in colab using cosine similarity

version2:  we created index on 1-mil pages and the all the terms from the Jason file
Added functionality:
1) We fixed many bugs and made the search work properly
2) Added another search function using MB25 did not test it yet.
3) Learned how to read the and write to/from the bucket
4) Did not run colab fronted yet, but results from one mil seems to not appear in the final answer.

version3: we created index on all pages and only the terms from the Jason file
Added functionality:
1) We overcame crushes by writing small portion of our main dictionaries to file; the index is created only on the terms from the Jason file
2) We fixed many bugs and checked, tested BM25.
3) We ran colab fronted, but something is still not right, result give 0 due to bad backend implementation.

version4: created a full index on all pages and all terms
Added functionality:
1) Applied stemming
3) tested with colab fronted: Average quality score: 0.25616666666666665, avg_duration:3.8
4) Added path option to writing/reading from/to the backend for dictionary operations in inverted gcp
5) Made a little bit optimizations to run time

vertion5: implemented read and write for PageRank
Added functionality:
1) Made many optimizations for time retrieval such as: sort first n, instead of sort m and take n.
2) Made optimizations for quality retrieval such as: boosting match for title and implementing PageRank
3) Tested many variation of parameters for BM25 and weight for title and PageRank

