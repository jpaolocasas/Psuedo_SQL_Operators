from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import csv
import logging
from typing import List, Tuple
import uuid
from functools import cmp_to_key
import os


#import ray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Generates unique operator IDs
def _generate_uuid():
    return uuid.uuid4()


# Custom tuple class with optional metadata
class ATuple:
    """Custom tuple.

    Attributes:
        tuple (Tuple): The actual tuple.
        metadata (string): The tuple metadata (e.g. provenance annotations).
        operator (Operator): A handle to the operator that produced the tuple.
    """
    def __init__(self, tuple, metadata=None, operator=None):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator

    # Returns the lineage of self
    def lineage(self) -> List[ATuple]:

        return self.operator.lineage([self])[0]

    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(self,att_index) -> List[Tuple]:

        return self.operator.where(att_index,[self])[0]

    # Returns the How-provenance of self
    def how(self) -> string:

        assert self.metadata is not None
        return self.metadata

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs(self) -> List[Tuple]:
        
        how_str=self.how()
        if how_str.startswith("GROUPBY"):
            avg_str = how_str[15:-1]
        else:
            avg_str = how_str

        total_clauses = []
        total_tuples={}
        for avg_clauses in avg_str.split(":"):
            trim_avg=avg_clauses[4:-1] #removes "AVG(" .... ")"
            coincident_tuples=trim_avg.split(',')   # split into fi*rj@x data ....
            how_parsed=[]
            for t in coincident_tuples:
                remove_val=t.split('@')
                tuple_ids=remove_val[0].split('*')  # will always generate 2 items
                total_tuples[tuple_ids[0]] = 0      # for tracking unique tuple ids ...
                total_tuples[tuple_ids[1]] = 0      # for tracking unique tuple ids ...
                how_parsed.append((tuple_ids,remove_val[1]))
            total_clauses.append(how_parsed)

        uniq_tuples=list(total_tuples.keys())

        # return True if removing t or rho will generate a different result.
        # this is specific to the query for assignment 2, task 4.
        def changed_result(t, rho=None):
            if rho is None:
                rho=t
            # clauses are already sorted from OrderBy so entry [0] has the highest rating.
            # to check if removing t and rho generates a different result, only need to
            # check if some other clause generates a higher rating than entry [0].
            r_avg = None
            for c in total_clauses:
                new_c = [j[1] for j in c if (t not in j[0]) and (rho not in j[0])]
                avg = float(AVG(new_c)) if new_c != [] else -1
                if r_avg is None:
                    r_avg = avg   # clause[0] avg is the rating of the original query result
                if avg > r_avg:
                    return True   # found an avg > r_avg so query has a different result
            return False

        responsibility = []
        for t in uniq_tuples:
            res = 0.0
            if changed_result(t):
                res = 1.0
            else:
                # since only looking for responsibility >= 0.5, only need to look for rho
                # of size 1. 
                for rho in uniq_tuples:
                    if rho != t and changed_result(t, rho):
                        res = 0.5
                        break
            if res >= 0.5:
                responsibility.append((t, res))
        return responsibility


    def __repr__(self):
        return str(self.tuple)

# Data operator
class Operator:
    """Data operator (parent class).

    Attributes:
        id (string): Unique operator ID.
        name (string): Operator name.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    def __init__(self, id=None, name=None, track_prov=False,
                                           propagate_prov=False):
        self.id = _generate_uuid() if id is None else id
        self.name = "Undefined" if name is None else name
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        logger.debug("Created {} operator with id {}".format(self.name,
                                                             self.id))

    # NOTE (john): Must be implemented by the subclasses
    def get_next(self):
        logger.error("Method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def lineage(self, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Lineage method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def where(self, att_index: int, tuples: List[ATuple]) -> List[List[Tuple]]:
        logger.error("Where-provenance method not implemented!")

# Scan operator
class Scan(Operator):
    """Scan operator.

    Attributes:
        filepath (string): The path to the input file.
        filter (function): An optional user-defined filter.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes scan operator
    def __init__(self, filepath,batch_size=100, filter=None, track_prov=False,
                                              propagate_prov=False):
        # super(Scan, self).__init__(name="Scan", track_prov=track_prov,
        #                            propagate_prov=propagate_prov)
        super().__init__(name="Scan", track_prov=track_prov,
                                   propagate_prov=propagate_prov)
        self.filepath=filepath
        self.filename=os.path.basename(filepath)
        self.batch_size=batch_size
        self.curr_line=1
        self.filter = filter
        self.file=open(self.filepath,'r',newline='\n')

        self.reader=csv.reader(self.file,delimiter=' ')     
        self.header=next(self.reader) #skip header

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):

        if self.file is None:
            return None
        data_batch=[0]*self.batch_size
        count=0
        for i in range(self.batch_size):
            try:
                data=tuple(next(self.reader))
                #meta_str=self.filename+":"+str(self.curr_line)
                # assumes filenames will not start with the same character!!
                meta_str=None
                if self.propagate_prov:
                    meta_str=(self.filename[0].lower())+str(self.curr_line)
                data_batch[count]=ATuple(data,meta_str,self) 
                count+=1
                self.curr_line+=1
            except StopIteration:
                self.file.close()
                self.file = None
                if count==0:
                    return None
                data_batch=data_batch[:count]
                break
        return data_batch

        
    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        return [[t] for t in tuples]

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):

        where_list=[]
        for t in tuples:
            #[fileName,lineNum]=t.how().split(":")
            lineNum=t.how()[1:]  # metadata has a first character id and then line number
            where_list.append((self.filename,lineNum,t.tuple,t.tuple[att_index]))
        return where_list
# Equi-join operator
class Join(Operator):
    """Equi-join operator.

    Attributes:
        left_input (Operator): A handle to the left input.
        right_input (Operator): A handle to the left input.
        left_join_attribute (int): The index of the left join attribute.
        right_join_attribute (int): The index of the right join attribute.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes join operator
    def __init__(self, left_input, right_input, left_join_attribute,
                                                right_join_attribute,
                                                track_prov=False,
                                                propagate_prov=False):
        super().__init__(name="Join", track_prov=track_prov,
                                   propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.left_input=left_input
        self.right_input=right_input
        self.left_join_attribute=left_join_attribute
        self.right_join_attribute=right_join_attribute
        self.pushed_forward={}
        self.left_hash=None
        self.left_len=0
        
    # Returns next batch of joined tuples (or None if done)
    def get_next(self):

        if self.left_hash is None:
            self.left_hash={}
            while True:
                left_data=self.left_input.get_next()
                if left_data is None:
                    break
                for d in left_data:
                    left_key=d.tuple[self.left_join_attribute]
                    if left_key not in self.left_hash: 
                        self.left_hash[left_key]=[]
                    self.left_hash[left_key].append(d)
                self.left_len=len(d.tuple)

        join_data=[]
        join_data_dict={}
        while True:   # get next data with matching keys
            right_data=self.right_input.get_next()
            if right_data is None:
                return None
        
            for d in right_data:
                right_key = d.tuple[self.right_join_attribute]
                if right_key in self.left_hash:
                    # join_data.extend([ATuple(leftk.tuple+d.tuple,None,self) for leftk in self.left_hash[right_key]])
                    for leftk in self.left_hash[right_key]:
                        meta_str=None
                        if self.propagate_prov:
                            meta_str=leftk.how()+"*"+d.how()
                        curr_tuple=ATuple(leftk.tuple+d.tuple,meta_str,self)
                        join_data.append(curr_tuple)
                        if self.track_prov:
                            self.pushed_forward[curr_tuple]=[leftk,d]
                    
            return join_data


    # Returns the lineage of the given tuples
    def lineage(self, tuples):

        def single_lineage(tuple):
            assert tuple in self.pushed_forward
            lin = []
            uniq = {}
            for t in self.pushed_forward[tuple]:
                # lin.extend(t.lineage())
                for t_lineage in t.lineage():
                    if t_lineage.tuple not in uniq:  # check for duplicates 
                        lin.append(t_lineage)
                        uniq[t_lineage.tuple]=True
            return lin        
        return [ single_lineage(t) for t in tuples if t in self.pushed_forward ]
        
        

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):

        def single_where(tuple):
            assert tuple in self.pushed_forward
            if att_index<self.left_len:
                return self.pushed_forward[tuple][0].where(att_index)
            else:
                return self.pushed_forward[tuple][1].where(att_index-self.left_len)
        
        return [single_where(t) for t in tuples if t in self.pushed_forward]
                


# Project operator
class Project(Operator):
    """Project operator.

    Attributes:
        input (Operator): A handle to the input.
        fields_to_keep (List(int)): A list of attribute indices to keep.
        If empty, the project operator behaves like an identity map, i.e., it
        produces and output that is identical to its input.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes project operator
    def __init__(self, input, fields_to_keep=[], track_prov=False,
                                                 propagate_prov=False):
        super().__init__(name="Project", track_prov=track_prov,
                                      propagate_prov=propagate_prov)

        self.input=input 
        self.fields_to_keep=fields_to_keep
        self.pushed_forward={}

    # Return next batch of projected tuples (or None if done)
    def get_next(self):

        data=self.input.get_next()
        if data is None:
            return None
        if self.fields_to_keep==[]:
            self.fields_to_keep = list(range(len(data)))
        project_data=[]
        for d in data:
            data_to_keep = [d.tuple[f] for f in self.fields_to_keep]
            meta_str=None
            if self.propagate_prov:
                meta_str = d.how()
            curr_tuple=ATuple(tuple(data_to_keep),meta_str,self)
            project_data.append(curr_tuple)
            if self.track_prov:
                self.pushed_forward[curr_tuple]=d
        return project_data 

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
      
        return [ self.pushed_forward[t].lineage() for t in tuples if t in self.pushed_forward ]

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):

        return [self.pushed_forward[t].where(self.fields_to_keep[att_index]) 
                    for t in tuples if t in self.pushed_forward]

# Group-by operator
class GroupBy(Operator):
    """Group-by operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples.
        value (int): The index of the attribute we want to aggregate.
        agg_fun (function): The aggregation function (e.g. AVG)
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes average operator
    def __init__(self, input, key, value, agg_fun, track_prov=False,
                                                   propagate_prov=False):
        super().__init__(name="GroupBy", track_prov=track_prov,
                                      propagate_prov=propagate_prov)
        self.input=input 
        self.key=key 
        self.value=value 
        self.agg_fun=agg_fun

        self.group_hash=None
        self.pushed_forward={}


    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):

        if self.group_hash is None:
            self.group_hash = {}
            while True:
                data=self.input.get_next()
                if data is None:
                    break
                for d in data:
                    group_key=d.tuple[self.key]
                    if group_key not in self.group_hash:
                        self.group_hash[group_key]=[]
                    self.group_hash[group_key].append(d) 

            self.currentKeys=list(self.group_hash.keys())
            self.currentKey=0

        if self.currentKey==len(self.currentKeys):
            return None

        key_list=[]   # can batch returns ... but return one result at a time for now

        key=self.currentKeys[self.currentKey]
        self.currentKey+=1

        agg_values = []
        meta_values = []
        for t in self.group_hash[key]:
            val = t.tuple[self.value]
            agg_values.append(val)
            if self.propagate_prov:
                meta_values.append(t.how()+"@"+val)
        agg_val=self.agg_fun(agg_values)
        meta_str=None
        if self.propagate_prov:
            meta_str=self.agg_fun.__name__+"("+ (",".join(meta_values)) +")"

        curr_tuple=ATuple((key,agg_val),meta_str,self)
        key_list.append(curr_tuple)
        if self.track_prov:
            self.pushed_forward[curr_tuple]=self.group_hash[key]

        return key_list


    # Returns the lineage of the given tuples
    def lineage(self, tuples):

        def single_lineage(tuple):
            lin = []
            uniq = {}
            for t in self.pushed_forward[tuple]:
                # lin.extend(t.lineage())
                for t_lineage in t.lineage():
                    if t_lineage.tuple not in uniq:  # check for duplicates 
                        lin.append(t_lineage)
                        uniq[t_lineage.tuple]=True
            return lin
        return [ single_lineage(t) for t in tuples if t in self.pushed_forward ]


    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):

        assert att_index==1 or att_index==0
        def single_where(tuple):
            res = []
            for t in self.pushed_forward[tuple]:
                res.append(t.where(self.key if att_index==0 else self.value))        
            return res
        return [ single_where(t) for t in tuples if t in self.pushed_forward ]


# Custom histogram operator
class Histogram(Operator):
    """Histogram operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples. The operator outputs
        the total number of tuples per distinct key.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes histogram operator
    def __init__(self, input, key=0, track_prov=False, propagate_prov=False):
        super().__init__(name="Histogram",
                                        track_prov=track_prov,
                                        propagate_prov=propagate_prov)

        self.input=input 
        self.key=key
        self.getNextCalled = False

    # Returns histogram (or None if done)
    def get_next(self):

        if self.getNextCalled:
            return None

        ratingDict={}
        while True:
            data=self.input.get_next()
            if data is None:
                break
            for d in data:
                rating=d.tuple[self.key]
                ratingDict[rating]=ratingDict.get(rating,0)+1     

        hist = [ATuple((k, str(v)),None,self) for k,v in ratingDict.items()]
        self.getNextCalled = True
        return hist


# Order by operator
class OrderBy(Operator):
    """OrderBy operator.

    Attributes:
        input (Operator): A handle to the input
        comparator (function): The user-defined comparator used for sorting the
        input tuples.
        ASC (bool): True if sorting in ascending order, False otherwise.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes order-by operator
    def __init__(self, input, comparator, ASC=True, track_prov=False,
                                                    propagate_prov=False):
        super().__init__(name="OrderBy",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov)

        self.input=input
        self.comparator=comparator
        self.ASC=ASC
        self.tuple_list=[]
        self.pushed_forward={}
        self.getNextCalled=False

    # Returns the sorted input (or None if done)
    def get_next(self):

        if self.getNextCalled==False:
            while True:
                data=self.input.get_next()
                if data is None:
                    break
                for d in data:
                    if self.track_prov:
                        self.pushed_forward[d.tuple]=d #hashing by t.tuple for comparator
                    self.tuple_list.append(d.tuple)

            data=sorted(self.tuple_list,key=cmp_to_key(self.comparator),reverse= not self.ASC)

            if self.propagate_prov and self.track_prov:
                # need to describe everything that gets sorted for responsibility computation
                # (to know if the sorting generates a different result)
                if len(data)>1:
                    meta_str=":".join(self.pushed_forward[d].how() for d in data)
                    dir_str="ASC" if self.ASC else "DESC"
                    data=[ATuple(d,f"GROUPBY({dir_str},{i},{meta_str})",self) for i,d in enumerate(data)] 
                else:
                    data=[ATuple(i,self.pushed_forward[i].how(),self) for i in data] 
            else:
                data=[ATuple(i,None,self) for i in data]
            self.getNextCalled=True
            return data
        return None

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        return [ self.pushed_forward[t.tuple].lineage() 
                    for t in tuples if t.tuple in self.pushed_forward ]

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        return [self.pushed_forward[t.tuple].where(att_index)
                    for t in tuples if t.tuple in self.pushed_forward ]

# Top-k operator
class TopK(Operator):
    """TopK operator.

    Attributes:
        input (Operator): A handle to the input.
        k (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes top-k operator
    def __init__(self, input, k=None, track_prov=False, propagate_prov=False):
        super().__init__(name="TopK", track_prov=track_prov,
                                   propagate_prov=propagate_prov)

        self.input=input 
        self.k=k
        self.itemsReturned=0
        self.pushed_forward={}

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):

        if self.itemsReturned==self.k:
            return None
        items=[]
        while True:
            data = self.input.get_next()#returns None if there are less then k items
            if data is None:
                self.itemsReturned=self.k #forces a None returned on next call
                break
            for d in data:
                if self.propagate_prov:
                    curr_tuple=ATuple(d.tuple,d.how(),self)
                else:
                    curr_tuple=ATuple(d.tuple,None,self)
                items.append(curr_tuple)
                if self.track_prov:
                    self.pushed_forward[curr_tuple]=d
                self.itemsReturned+=1
                if self.itemsReturned == self.k:
                    break
        return items


    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        return [ self.pushed_forward[t].lineage() 
                    for t in tuples if t in self.pushed_forward ]

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        return [self.pushed_forward[t].where(att_index) 
                    for t in tuples if t in self.pushed_forward ]

# Filter operator
class Select(Operator):
    """Select operator.

    Attributes:
        input (Operator): A handle to the input.
        predicate (function): The selection predicate.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes select operator
    def __init__(self, input, predicate, track_prov=False,
                                         propagate_prov=False):
        super().__init__(name="Select", track_prov=track_prov,
                                     propagate_prov=propagate_prov)
        self.predicate=predicate
        self.input=input
        self.pushed_forward={}

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):

        batch_data=[]
        while True:
            data=self.input.get_next()
            if data is None:
                return None
            for d in data:
                valid_data=self.predicate(d.tuple)
                if valid_data:
                    if self.propagate_prov:
                        curr_tuple=ATuple(d.tuple,d.how(),self)
                    else:
                        curr_tuple=ATuple(d.tuple,None,self)
                    batch_data.append(curr_tuple)
                    if self.track_prov:
                        self.pushed_forward[curr_tuple]=d
            if batch_data!=[]:
                return batch_data

    def lineage(self,tuples):
        return [ self.pushed_forward[t].lineage() 
                    for t in tuples if t in self.pushed_forward ]

    def where(self,att_index,tuples):
        return [self.pushed_forward[t].where(att_index) 
                    for t in tuples if t in self.pushed_forward ]



def gen_predicate(index, value):
    def pred(data):
        return data[index]==value
    return pred

def AVG(data):
    sum=0
    if data==[]:
        return None
    for d in data:
        sum+=int(d[0])

    avg=sum/len(data)
    return str(avg)

def gen_compare(index):
    def compare(tuple1,tuple2):
        if float(tuple1[index])<float(tuple2[index]):
            return -1
        elif float(tuple1[index])>float(tuple2[index]):
            return 1
        else:
            return 0
    return compare

def getEntireInput(input):
    data_to_keep=[]
    while True:
        data=input.get_next()
        if data is None:
            break
        for d in data:
            # data_to_keep.append(d.tuple)
            data_to_keep.append(d)

    return data_to_keep

import sys
if __name__ == "__main__":
    
    task=None
    pathToFriends=None
    pathToRatings=None
    batch_size=10
    userID=None
    movieID=None
    assignment=None

    for i in range(1, len(sys.argv), 2):
        if sys.argv[i]=="--task":
            task = int(sys.argv[i+1])
        elif sys.argv[i]=="--friends":
            pathToFriends = sys.argv[i+1]
        elif sys.argv[i]=="--ratings":
            pathToRatings = sys.argv[i+1]
        elif sys.argv[i]=="--batch_size":
            batch_size = int(sys.argv[i+1])
        elif sys.argv[i]=="--uid":
            userID = sys.argv[i+1]
        elif sys.argv[i]=="--mid":
            movieID = sys.argv[i+1]
        elif sys.argv[i]=="--assignment":
            assignment = int(sys.argv[i+1])
        else:
            print(f"Invalid argument {sys.argv[i]}")
            exit()

    def check(msg, val):
        if val is None:
            print(f"{msg} not specified")
            exit()
        else:
            print(f"Got {msg} = {val}")

    check("--task", task)
    check("--friends", pathToFriends)
    check("--ratings", pathToRatings)
    check("--batch_size", batch_size)
    check("--uid", userID)
    check("--assignment", assignment)
    if assignment==1:
        if task in [1,3,4]:
            check("--mid", movieID)
    elif assignment==2:
        # task=task+4
        if task in [2]:
            check("--mid", movieID)

    # print(task,batch_size,userID,movieID)
    logger.info("Assignment #1")    

    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    def task1(pathToFriends,pathToRatings,batch_size,userID,movieID):
        R=Scan(pathToRatings,batch_size)
        F=Scan(pathToFriends,batch_size)
        F_select=Select(F, gen_predicate(0, userID))  # 0=F.UID1 index
        R_select=Select(R, gen_predicate(1, movieID))     # 1=R.MID index
        combined=Join(F_select,R_select, 1, 0)        # 1=F.UID2, 0=R.UID
        ratings=Project(combined,[4])                 # 4=Ratings in joined tuple
        ratings_list=getEntireInput(ratings)
        # print(ratings_list)
        if ratings_list==[]:
            print("Query returned no results")
        else:
            avg=AVG([r.tuple for r in ratings_list])
            print("Average rating: ",avg)

    # TASK 2: Implement recommendation query for User A
    #
    # SELECT R.MID
    # FROM ( SELECT R.MID, AVG(R.Rating) as score
    #        FROM Friends as F, Ratings as R
    #        WHERE F.UID2 = R.UID
    #              AND F.UID1 = 'A'
    #        GROUP BY R.MID
    #        ORDER BY score DESC
    #        LIMIT 1 )

    def task2(pathToFriends,pathToRatings,batch_size,userID):
        R=Scan(pathToRatings,batch_size)
        F=Scan(pathToFriends,batch_size)
        F_select=Select(F, gen_predicate(0,userID)) #all the friends of person 1190
        combined=Join(F_select,R,1,0) #WHERE F.UID2=R.UID, all the movies rated by the friends of 1190
        agg_avgs=GroupBy(combined,3,4,AVG)
        ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False)
        movieSuggestion=TopK(ordered_avgs,1)

        proj_output=Project(movieSuggestion,[0]) #return top k(1) movie ID,0=movieID
        output=getEntireInput(proj_output)
        print(output)   


    # TASK 3: Implement explanation query for User A and Movie M
    #
    # SELECT HIST(R.Rating) as explanation
    # FROM Friends as F, cvcvRatings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    def task3(pathToFriends,pathToRatings,batch_size,userID,movieID):
        R=Scan(pathToRatings,batch_size)
        F=Scan(pathToFriends,batch_size)
        F_select=Select(F, gen_predicate(0,userID)) #all the friends of person 1190
        R_select=Select(R, gen_predicate(1,movieID)) 
        combined=Join(F_select,R_select, 1, 0)
        counts=Histogram(combined,4)
        output=getEntireInput(counts)

        for data in output:
            print(data)

    if assignment==1:
        if task==1:
            task1(pathToFriends,pathToRatings,batch_size,userID,movieID)
        elif task==2:
            task2(pathToFriends,pathToRatings,batch_size,userID)
        elif task==3:
            task3(pathToFriends,pathToRatings,batch_size,userID,movieID)

    logger.info("Assignment #2")

    # TASK 1: Implement lineage query for movie recommendation

    def check_lineage(x):
        output=getEntireInput(x)
        for i,o in enumerate(output):
            print(f"lineage {i} {o} {o.lineage()}")
        raise Exception("check_lineage break")

    def check_where(x, att_index):
        output=getEntireInput(x)
        for i,o in enumerate(output):
            print(f"where({att_index}) {i} {o} {o.where(att_index)}")
        raise Exception("check_where break")

    def check_how(x):
        output=getEntireInput(x)
        for i,o in enumerate(output):
            print(f"how {i} {o} {o.how()}")
        raise Exception("check_how break")

    def task_A2_1(pathToFriends,pathToRatings,batch_size,userID):
        track_prov=True
        propagate_prov=False
        R=Scan(pathToRatings,batch_size,None,track_prov,propagate_prov)
        F=Scan(pathToFriends,batch_size,None,track_prov,propagate_prov)
        F_select=Select(F, gen_predicate(0, userID),track_prov,propagate_prov)  # 0=F.UID1 index
        combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
        agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
        ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
        movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
        proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
        output=getEntireInput(proj_output)
        print("Lineage: ",output[0].lineage())

    # TASK 2: Implement where-provenance query for 'likeness' prediction

    def task_A2_2(pathToFriends,pathToRatings,batch_size,userID,movieID):
        track_prov=True
        propagate_prov=False  # only scan needs propage_prov to track line numbers.
        R=Scan(pathToRatings,batch_size,None,track_prov,True)
        F=Scan(pathToFriends,batch_size,None,track_prov,True)
        F_select=Select(F, gen_predicate(0, userID),track_prov,propagate_prov)  # 0=F.UID1 index
        R_select=Select(R, gen_predicate(1, movieID),track_prov,propagate_prov)     # 1=R.MID index
        combined=Join(F_select,R_select, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
        agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
        proj_output=Project(agg_avgs,[1],track_prov,propagate_prov)
        output=getEntireInput(proj_output)
        print(output[0].where(0))

    # TASK 3: Implement how-provenance query for movie recommendation

    def task_A2_3(pathToFriends,pathToRatings,batch_size,userID):
        track_prov=True
        propagate_prov=True
        R=Scan(pathToRatings,batch_size,None,track_prov,propagate_prov)
        F=Scan(pathToFriends,batch_size,None,track_prov,propagate_prov)
        F_select=Select(F, gen_predicate(0, userID),track_prov,propagate_prov)  # 0=F.UID1 index
        combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
        agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
        ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
        movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
        proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
        output=getEntireInput(proj_output)
        print(output[0].how())

    # TASK 4: Retrieve most responsible tuples for movie recommendation

    def task_A2_4(pathToFriends,pathToRatings,batch_size,userID):
        track_prov=True
        propagate_prov=True
        R=Scan(pathToRatings,batch_size,None,track_prov,propagate_prov)
        F=Scan(pathToFriends,batch_size,None,track_prov,propagate_prov)
        F_select=Select(F, gen_predicate(0, userID),track_prov,propagate_prov)  # 0=F.UID1 index
        combined=Join(F_select,R, 1, 0,track_prov,propagate_prov)        # 1=F.UID2, 0=R.UID
        agg_avgs=GroupBy(combined,3,4,AVG,track_prov,propagate_prov)
        ordered_avgs=OrderBy(agg_avgs,gen_compare(1),False,track_prov,propagate_prov)
        movieSuggestion=TopK(ordered_avgs,1,track_prov,propagate_prov)
        proj_output=Project(movieSuggestion,[0],track_prov,propagate_prov)
        output=getEntireInput(proj_output)
        print(output[0].responsible_inputs())   

    if assignment==2:
        if task==1: #lineage
            task_A2_1(pathToFriends,pathToRatings,batch_size,userID)
        elif task==2: #where-provenance
            task_A2_2(pathToFriends,pathToRatings,batch_size,userID,movieID)
        elif task==3: #how-provenance
            task_A2_3(pathToFriends,pathToRatings,batch_size,userID)
        elif task==4: #responsible tuples   
            # print("HERE")
            task_A2_4(pathToFriends,pathToRatings,batch_size,userID)
