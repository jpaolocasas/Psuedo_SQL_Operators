# Psuedo_SQL_Operators
Code written for the User-Center Systems for Data Science (CS599 Fall 2021) course at Boston University with instructor John Liagouris. Implemented SQL operators from scratch in Python to mimic a recommendation system using movie relational databases.

## Input Data

Queries of assignments 1 and 2 expect two space-delimited text files (similar to CSV files). 

The first file (friends list) must include records of the form:

|UID1 (int)|UID2 (int)|
|----|----|
|1   |2342|
|231 |3   |
|... |... |

In this database, UID1 and UID2 are the unique ID's of distinct people with information in the database. Since there exists a row with UID1: 1 and UID2: 2342, this means that users 1 and 2342 are friends.

The second file (ratings) must include records of the form:

|UID (int)|MID (int)|RATING (int)|
|---|---|------|
|1  |10 |4     |
|231|54 |2     |
|...|...|...   |

In this database, UID represents the unique ID of the user, MID stands for the unique movie ID, and RATING is the rating that the user gave the movie (scale from 1 to 5). For example, the first row states that: user 1 watched movie 10 and rated it a 4 out of 5.

## Queries

There are three types of queries implemented:

The first a likeness prediction for user A and movie M based on the ratings of movie M given by the friends of A.
In SQL this would look like:
```
SELECT AVG(R.Rating)
     FROM Friends as F, Ratings as R
     WHERE F.UID2 = R.UID
           AND F.UID1 = 'A'
           AND R.MID = 'M'
```

The second is a movie recommendation query based on the highest likeness value amongst the user's friends. Namely, for all the users that are friends of user A, compute the average score of every movie they watched and return the ID of the movie with the highest score.
In SQL this would look like:
```
 SELECT R.MID
     FROM ( SELECT R.MID, AVG(R.Rating) as score
            FROM Friends as F, Ratings as R
            WHERE F.UID2 = R.UID
                  AND F.UID1 = 'A'
            GROUP BY R.MID
            ORDER BY score DESC
            LIMIT 1 )
```

The third is a basic explanation query for the recommended movie in the second query using a histogram.
In SQL this would look like:
```
 SELECT HIST(R.Rating) as explanation
     FROM Friends as F, Ratings as R
     WHERE F.UID2 = R.UID
           AND F.UID1 = 'A'
           AND R.MID = 'M'
 ```
 
 ## Class Methods
 Each operator implemented is a class with the following methods init(), get_next(), lineage(), and where().
 
 The init() method simply creates an instance of the operator and handles any initalization steps before a request for data is made.
 
 The get_next() method is used to grab the next batch of data for processing. The system is pull-based so by sending the data in batches, unnecessary memory usage is avoided. This pipelined fashion of sending data in batches is favorable when compared to the alternative of loading the entire input file in memory. Since the system is pull-based, the get_next() method also handles most of the "operator" behavior as well.
 
 The lineage() method is used to get the collection of input tuples (raw data) that contributed to the output of the query.
 
 The where() method builds upon the lineage() method and rather than returning just the value input tuple itself, it also returns data regarding the specific input file and line number the input tuple originiated from. The lineage() and where() methods required back-tracking techniques and some metadata to be saved at each stage in the query but allow for higher output explanability in return.
 
 ## Additional Methods
 Lastly, there are the how() and responsible_inputs() methods that are designed to be used on the output of an entire query (as opposed to the lineage() and where() methods that can be used at any point in the data pipeline).
 
 The how() method provides information regarding how the output was computed by returning a string containing pairs of unique identifiers which co-exist in the input files, the attribute value (movie rating) associated with the pairs, and the operation (such as AVG) which computed the output based on the attribute value.
 
 As the name suggests, the responsible_inputs() method returns the input tuples that contributed 50% or more to the ouput.
