from assignment_12 import *
import pytest

# Assignment 1 Tests
# ---------------------------------------------------------------------------------------
pathToFriends="../data/test_friends.txt"
pathToRatings="../data/test_ratings.txt"


# Scan Tests
# -------------------------------------------------------------
# using default batch size (no argument passed)
# try to scan a file that doesn't exist
# batch size bigger than file
# batch size smaller than file
def testScanFriends_defaultBatch():
	f=Scan(pathToFriends)
	assert len(getEntireInput(f)) == 14

def testScanFriends_defaultBatch_contents():
	f=Scan(pathToFriends)
	output=getEntireInput(f)
	assert output[0].tuple==('0','1') and output[-1].tuple==('3','2')

def testScanFriends_defaultBatch_size():
	f=Scan(pathToFriends)
	output=getEntireInput(f)
	for o in output:
		assert len(o.tuple)==2

def testScanRatings_defaultBatch():
	f=Scan(pathToRatings)
	assert len(getEntireInput(f)) == 16

def testScanRatings_defaultBatch_contents():
	f=Scan(pathToRatings)
	output=getEntireInput(f)
	assert output[0].tuple==('0','0','0') and output[-1].tuple==('4','4','4')

def testScanRatings_defaultBatch_size():
	f=Scan(pathToRatings)
	output=getEntireInput(f)
	for o in output:
		assert len(o.tuple)==3

batch_size1=2 #smaller than file
batch_size2=1000 #bigger than file, default of 100 is already bigger but just to check

def testScanFriends_smallBatch():
	f=Scan(pathToFriends,batch_size1)
	assert len(getEntireInput(f))==14

def testScanRatings_bigBatch():
	f=Scan(pathToFriends,batch_size2)
	assert len(getEntireInput(f))==14

# # Select Tests
# # -------------------------------------------------------------
# # normal select x2
# # item doens't exist in select --> expect an error 

def testSelectFriends1():
	friends_scan=Scan(pathToFriends)
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	output=getEntireInput(f_select)
	assert output[0].tuple==('0','1') and output[-1].tuple==('0','4')

def testSelectFriends2():
	friends_scan=Scan(pathToFriends)
	f_select=Select(friends_scan,gen_predicate(0, '4'))
	output=getEntireInput(f_select)
	assert output[0].tuple==('4','0') and output[-1].tuple==('4','1')

def testSelectFriends_invald():
	friends_scan=Scan(pathToFriends)
	f_select=Select(friends_scan,gen_predicate(0, '5'))
	assert getEntireInput(f_select)==[]

def testSelectRatings1():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(0, '0'))
	output=getEntireInput(r_select)
	assert output[0].tuple==('0','0','0') and output[-1].tuple==('0','3','4')

def testSelectRatings2():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(1, '4'))
	output=getEntireInput(r_select)
	assert output[0].tuple==('4','4','4')

def testSelectRatings_invalid():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(1, '5'))
	assert getEntireInput(r_select)==[]

def testSelectRatings_len():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(1, '4'))
	output = getEntireInput(r_select)
	for o in output:
		assert len(o.tuple)==3

# # Join Tests
# # -------------------------------------------------------------
# # works as expected, tuples retruned x2
# # works as expected, but no matches exist

def testJoin1():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '0'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	output=getEntireInput(combined)
	assert len(output)==4

def testJoin2():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '4'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	output=getEntireInput(combined)
	assert len(output)==1

def testJoin3():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '5'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	output=getEntireInput(combined)
	assert output==[]

def testJoin3():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '4'))
	f_select=Select(friends_scan,gen_predicate(1, '2'))
	combined=Join(f_select,r_select,1,0)
	output=getEntireInput(combined)
	assert output==[]

# # Project Tests
# # -------------------------------------------------------------
# # normal behavior x2
def testProject1():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '0'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	proj=Project(combined,[0])
	output=getEntireInput(proj)
	assert len(output)==4

def testProject2():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '0'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	proj=Project(combined,[0,1,2,3,4])
	output=getEntireInput(proj)
	for o in output:
		assert len(o.tuple)==5

def testProject3():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '4'))
	f_select=Select(friends_scan,gen_predicate(0, '0'))
	combined=Join(f_select,r_select,1,0)
	proj=Project(combined,[4])
	output=getEntireInput(proj)
	assert output[0].tuple==('4',)

def testProject4():
	ratings_scan=Scan(pathToRatings)
	friends_scan=Scan(pathToFriends)
	r_select=Select(ratings_scan,gen_predicate(1, '3'))
	f_select=Select(friends_scan,gen_predicate(0, '1'))
	combined=Join(f_select,r_select,1,0)
	proj=Project(combined,[3])
	output=getEntireInput(proj)
	for o in output:
		assert o.tuple==('3',)

# AVG
# -------------------------------------------------------------
# Empty list as input
# works as expected x2
inputNums1=[('0',),('0',),('3',)]
inputNums2=[('1',),('2',),('3',)]
inputNums3=[]

def testAVG1():
	assert AVG(inputNums1)=='1.0'

def testAVG2():
	assert AVG(inputNums2)=='2.0'

def testAVG3():
	assert AVG(inputNums3)==None
# GroupBy
# -------------------------------------------------------------
# works as expected (different keys to group by and values) x2
def testGroupBy1():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(1, '0'))
	grouped=GroupBy(r_select,0,1,AVG)
	output=getEntireInput(grouped)
	assert len(output)==5 and output[0].tuple==('0','0.0')

def testGroupBy2():
	ratings_scan=Scan(pathToRatings)
	r_select=Select(ratings_scan,gen_predicate(1, '1'))
	grouped=GroupBy(r_select,1,2,AVG)
	output=getEntireInput(grouped)
	assert len(output)==1 and output[0].tuple==('1','2.0')
# OrderBy
# -------------------------------------------------------------
# works as expected (ascend/descend, different index of interest)
def testOrderBy1():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(1),False)
	output=getEntireInput(ordered)
	assert output[0].tuple==('4','4','4') and len(output)==16

def testOrderBy2():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(2),False)
	output=getEntireInput(ordered)
	assert output[0].tuple==('4','4','4') and len(output)==16

def testOrderBy2():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(0))
	output=getEntireInput(ordered)
	for o in output[:3]:
		assert o.tuple[0]=='0'

# Top-K
# -------------------------------------------------------------
# works as expected x2
# try to return more than file/batch size
def testTopK1():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(1),False)
	top=TopK(ordered,1)
	output=getEntireInput(top)
	assert len(output)==1 and output[0].tuple==('4','4','4')

def testTopK2():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(1))
	top=TopK(ordered,3)
	output=getEntireInput(top)
	assert len(output)==3

def testTopK3():
	ratings_scan=Scan(pathToRatings)
	ordered=OrderBy(ratings_scan,gen_compare(1))
	top=TopK(ordered,101)
	output=getEntireInput(top)
	assert len(output)==16

# Hist
# -------------------------------------------------------------
# works as expected
def testHist1():
	ratings_scan=Scan(pathToRatings)
	hist=Histogram(ratings_scan,2)
	output=getEntireInput(hist)
	assert len(output)==5

def testHist1():
	ratings_scan=Scan(pathToRatings)
	hist=Histogram(ratings_scan,2)
	ordered=OrderBy(hist,gen_compare(0)) #added order operator so the hist will be easier to check
	output=getEntireInput(ordered)
	assert output[0].tuple==('0','6')

def testHist2():
	ratings_scan=Scan(pathToRatings)
	hist=Histogram(ratings_scan,1)
	ordered=OrderBy(hist,gen_compare(0))
	output=getEntireInput(ordered)
	assert output[-1].tuple==('4','1')
	
# Assignment 2 Tests
# ---------------------------------------------------------------------------------------

# Paths for tasks 1,3,4 (lineage, where, and how)
batch_size=10
pathToFriends_A2="../data/friends_A2.txt"
pathToRatings_A2="../data/ratings_A2.txt"
pathToFriends_A2_T4="../data/friends_A2_T4.txt"
pathToRatings_A2_T4="../data/ratings_A2_T4.txt"

# Lineage
# -------------------------------------------------------------
def testLineage1():
	track_prov=True
	propagate_prov=False
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	lin=output[0].lineage()
	assert isinstance(lin,list)
	assert len(lin) == 6
	assert isinstance(lin[0],ATuple)
	assert lin[0].tuple == ('0','1') and lin[-1].tuple == ('18','10','2')

def testLineage2():
	track_prov=False
	propagate_prov=False
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	try:
		lin=output[0].lineage()
		has_error=False
	except IndexError:
		has_error=True 
	assert has_error, "expecting IndexError"

def testLineage3():
	track_prov=True
	propagate_prov=False
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	lin=output[0].lineage()
	assert isinstance(lin,list)
	assert len(lin) == 4
	assert isinstance(lin[0],ATuple)
	assert lin[-1].tuple==('2','10','3')

# Where-Prov
# -------------------------------------------------------------
def testWhere1():
	track_prov=True
	propagate_prov=False  # only scan needs propage_prov to track line numbers
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,True)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,True)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	R_select=Select(R, gen_predicate(1, '10'),track_prov,propagate_prov)     # 1=R.MID index
	combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	where_prov=output[0].where(0)
	assert isinstance(where_prov,list)
	assert len(where_prov)==3
	assert isinstance(where_prov[0],tuple)
	assert where_prov[0]==('ratings_A2.txt','1',('1','10','5'),'5')

def testWhere2():
	track_prov=False
	propagate_prov=False  # only scan needs propage_prov to track line numbers
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,True)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,True)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	R_select=Select(R, gen_predicate(1, '10'),track_prov,propagate_prov)     # 1=R.MID index
	combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	try:
		where_prov=output[0].where(0)
		has_error=False
	except IndexError:
		has_error=True 
	assert has_error, "expecting IndexError"

def testWhere3():
	track_prov=True
	propagate_prov=False  # only scan needs propage_prov to track line numbers
	try:
		R=Scan(pathToRatings_A2,batch_size,None,track_prov,False)
		F=Scan(pathToFriends_A2,batch_size,None,track_prov,False)
		F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
		R_select=Select(R, gen_predicate(1, '10'),track_prov,propagate_prov)     # 1=R.MID index
		combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
		agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
		proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
		output=getEntireInput(proj_output)

		where_prov=output[0].where(0)
		has_error=False
	except AssertionError:
		has_error=True 
	assert has_error, "expecting AssertionError"

def testWhere4():
	track_prov=True
	propagate_prov=False  # only scan needs propage_prov to track line numbers
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,True)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,True)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	R_select=Select(R, gen_predicate(1, '10'),track_prov,propagate_prov)     # 1=R.MID index
	combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	where_prov=output[0].where(0)
	assert isinstance(where_prov,list)
	assert len(where_prov)==2
	assert isinstance(where_prov[0],tuple)
	assert where_prov[-1]==('ratings_A2_T4.txt','2',('2','10','3'),'3')

def testWhere5():
	track_prov=True
	propagate_prov=False  # only scan needs propage_prov to track line numbers
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,True)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,True)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	R_select=Select(R, gen_predicate(1, '9'),track_prov,propagate_prov)     # 1=R.MID index
	combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	where_prov=output[0].where(0)
	assert isinstance(where_prov,list)
	assert len(where_prov)==2
	assert isinstance(where_prov[0],tuple)
	assert where_prov[0]==('ratings_A2_T4.txt','3',('3','9','4'),'4')

# How-Prov
# -------------------------------------------------------------
def testHow1():
	track_prov=True
	propagate_prov=True
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	how_prov=output[0].how()
	assert isinstance(how_prov,str)
	assert how_prov=="AVG(f1*r1@5,f2*r2@8,f3*r3@2)"

def testHow2():
	track_prov=True
	propagate_prov=False
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	
	try:
		how_prov=output[0].how()
		has_error=False
	except AssertionError:
		has_error=True
	assert has_error, "expecting AssertionError"

def testHow3():
	track_prov=True
	propagate_prov=True
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	how_prov=output[0].how()
	assert how_prov=="GROUPBY(DESC,0,AVG(f1*r1@5,f2*r2@3):AVG(f3*r3@4,f4*r4@2))"

# Responsibility
# -------------------------------------------------------------
def testRes1():
	track_prov=True
	propagate_prov=True
	R=Scan(pathToRatings_A2,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	res_inputs=output[0].responsible_inputs()
	assert isinstance(res_inputs,list)
	assert res_inputs==[]

def testRes2():
	track_prov=True
	propagate_prov=True
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	res_inputs=output[0].responsible_inputs()
	assert isinstance(res_inputs,list)
	assert len(res_inputs)==6
	for r in res_inputs:
		assert r[1]==0.5
	assert res_inputs[0]==('f1',0.5)

def testRes3():
	track_prov=False
	propagate_prov=True

	try:
		R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
		F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
		F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
		combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
		agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
		ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
		movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
		proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
		output=getEntireInput(proj_output)
		
		res_inputs=output[0].responsible_inputs()
		has_error=False
	except AssertionError:
		has_error=True
	assert has_error, "expecting AssertionError"

def testRes4():
	track_prov=True
	propagate_prov=False
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	try:
		res_inputs=output[0].responsible_inputs()
		has_error=False
	except AssertionError:
		has_error=True
	assert has_error, "expecting AssertionError"

def testRes5():
	track_prov=False
	propagate_prov=False
	R=Scan(pathToRatings_A2_T4,batch_size,None,track_prov,propagate_prov)
	F=Scan(pathToFriends_A2_T4,batch_size,None,track_prov,propagate_prov)
	F_select=Select(F, gen_predicate(0, '0'),track_prov,propagate_prov)  # 0=F.UID1 index
	combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
	agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
	ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
	movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
	proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
	output=getEntireInput(proj_output)
	try:
		res_inputs=output[0].responsible_inputs()
		has_error=False
	except AssertionError:
		has_error=True
	assert has_error, "expecting AssertionError"
