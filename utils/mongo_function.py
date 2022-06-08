def count_num(collection):
    agg_results=collection.aggregate([
      {"$group": {  "_id": "",   "comment_num":{"$sum":1}  }},  
    ])
    for agg in agg_results:
        print("count_num: ",agg)

def agg_media(my_collection):
    agg_results=my_collection.aggregate([
        {"$group": {  "_id": "$media",   "count":{"$sum":1}  }},  
        {"$sort" : {  "count":-1 }  }   
    ])
    for agg in agg_results:
        print("count_num: ",agg)

def agg_category(my_collection):
    agg_results=my_collection.aggregate([
        {"$group": {  "_id": "$category",   "count":{"$sum":1}  }},  
        {"$sort" : {  "count":-1 }  }   
    ])
    agg_list=[]
    for agg in agg_results:
        agg_list.append(agg)
    return agg_list




#     preprocessing  ##################################################

# my_collection.delete_many({"media":"風傳媒"})
# my_collection.update_many({"category":"運動"},{"$set":{"category":"體育"}})
# my_collection.update_many({"category":"全球"},{"$set":{"category":"國際"}})

def target_train_document(my_collection):
    agg_list = agg_category(my_collection)
    agg_list=agg_list[0:6]
    train_category=[]
    for agg in agg_list:
        train_category.append(agg['_id'])
    print(train_category)

    count=0
    for mongo in my_collection.find({}):
        count+=1
        title = mongo['title']
        category=mongo['category']
        _id=mongo['_id']
        if category in train_category:
            my_collection.update_one({"_id":ObjectId(_id)},{"$set":{"train_target":True}})
        else:
            my_collection.update_one({"_id":ObjectId(_id)},{"$set":{"train_target":False}})
    
        print(category,title,end="\n\n")
    print(count)