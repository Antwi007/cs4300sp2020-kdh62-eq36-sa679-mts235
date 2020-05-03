from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import random


from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

project_name = "Project Re-search"
net_id = "Nana Antwi: nka32, Max Stallop: mls235, Edwin Quaye: eq36, Stephen Adusei Owusu: sa679, Kenneth Harlley: kdh62"

allergy_dict = {
    "Lactose intolerant": ["Milk", "Milk powder", "Cheese", "Butter", "Margarine", "yogurt", "cream", "ice cream"],
    "Eggs allergy": ["egg", "eggs"],
    "Tree nuts allergy": ["Brazil nuts", "almonds", "cashews", "macadamia nuts", "pistachios", "pine nuts", "walnuts", "nuts"],
    "Peanuts allergy": ["peanut", "peanuts", "nuts"],
    "Shellfish allergy": ["Shrimp", "prawns", "crayfish", "lobster", "squid", "scallops"],
    "Wheat allergy": ["wheat"],
    "Soy allergy": ["soy", "soybean"],
    "Fish allergy": ["fish"],
    "others": ["linseed", "sesame seed", "peach", "banana", "avocado", "kiwi fruit", "passion fruit", "celery", "garlic", "mustard seed", "aniseed", "chamomile"]
}

#   quer_desc = query_desc.lower()
#   quer_desc = str(set(quer_desc.split()))
#   descrip_list = []
#   # Stemming
#   ps = PorterStemmer()
#   word_tokens = word_tokenize(query_desc)
#   word_tokens_1 = [w for w in word_tokens if not w in stop_words_1]
#   stem_set = set([ps.stem(word) for word in word_tokens_1])

#  # Find the intersection between the query description and the food description
#   # and if it's greater than 0, then there's a match.
#   for food_item in nutr_out:
#       # Stem the descriptions in json file
#       longd = word_tokenize(food_item['Descrip'].lower())
#       set_longd = set([ps.stem(descp) for descp in longd])


def allergen_val(allergy_dict, allergy):
    output = {}
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    # for allergy in allergy_dict:
    ids = []
    descrip_list = intersect(
        allergy_dict[allergy][0], nutrients_data, True, 100)
    # output_allergy =  [desc["Descrip"] for data in output_data]
    # for food in nutrients_data:
    #     fointersect(allergy_dict[allergy][0], food):
    #         ids.append(food["Descrip"])
    # output[allergy] = descrip_list
    return descrip_list


def intersect_old(str1, list1):
    str1 = str1.split(",")
    str1 = [value.lower() for value in str1]
    list1 = [value.lower() for value in list1]
    output = [value for value in list1 if value in str1]
    if len(output) > 0:
        return True
    return False


def intersect(str1, nutr_out, normal, limit):
    descrip_list = []
    stop_words_1 = stop_words()
    str1 = str1.lower()

    ps = PorterStemmer()
    word_tokens = word_tokenize(str1)
    word_tokens_1 = [w for w in word_tokens if not w in stop_words_1]
    stem_set = set([ps.stem(word) for word in word_tokens_1])
    for food_item in nutr_out:
        longd = word_tokenize(food_item['Descrip'].lower())
        set_longd = set([ps.stem(descp) for descp in longd])
        if normal:
            if len(stem_set.intersection(set_longd)) > 0:
                if food_item not in descrip_list:
                    if food_item not in descrip_list:
                        descrip_list.append(food_item['Descrip'])
                        continue
        else:
            if len(stem_set.intersection(set_longd)) == 0:
                if food_item not in descrip_list:
                    if food_item not in descrip_list:
                        descrip_list.append(food_item['Descrip'])
                        if len(descrip_list) >= limit:
                            break
                        continue
    return descrip_list


def reverse_allergen(allergy_dict, allergy):
    output = {}
    f = open('nutrients.json',)
    nutrients_data = json.load(f)

    ids = []
    num = 0
    descrip_list = intersect(
        allergy_dict[allergy][0], nutrients_data, False, 100)
    # for food in nutrients_data:

    #     if not intersect(allergy_dict[allergy][0], food) and num != 100:
    #         ids.append(food["Descrip"])
    #         num += 1
    # output[allergy] = descrip_list

    return descrip_list


def ml_list():
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    non_lactose = []
    lactose = []
    for item in nutrients_data:
        # food_group = item["FoodGroup"]
        if item["FoodGroup"] == "Poultry Products":
            non_lactose.append(item["Descrip"])
        elif item["FoodGroup"] == "Dairy and Egg Products":
            lactose.append(item["Descrip"])
    f.close()
    return lactose, non_lactose


# returns the indices of output_data that satisfy the allergen


def bernoulli_nb(allergy_name, output_data):
    """
    Returns the indices of output data that based on the bernoulli naive-bayes algorithm
    won't contain any allergens.
    """
    # breaking some food item descriptions into lactose and non-lactose then noting their classes in seperate arrays with matching indexes
    # allergen is a list of descriptions, for a specified allergy

    output_data = [data["Descrip"] for data in output_data]
    # STEPHEN FUNCTION CHANGE
    allergen, non_allergen = reverse_allergen(
        allergy_dict, allergy_name), allergen_val(allergy_dict, allergy_name)
    # allergen, non_allergen = rv_allergens[allergy_name], pos_allergens[allergy_name]
    allergen_classes = [allergy_name for _ in allergen]
    non_allergen_classes = ["Not Allergen" for _ in non_allergen]
    descs = allergen + non_allergen
    descs = np.array(descs)
    classes = allergen_classes + non_allergen_classes
    classes = np.array(classes)

    # obtaining a random division of indexes to be able to get testing and training data
    nr_descs = len(descs)
    shuffle_split = ShuffleSplit(nr_descs, test_size=0.5, random_state=0)
    x = shuffle_split.split(descs)
    for train_idx, test_idx in x:
        pass

    # training and test data
    descs_train = descs[train_idx]
    descs_test = descs[test_idx]
    classes_train = classes[train_idx]
    classes_test = classes[test_idx]

    # obtain term document matrix which will be used to fit the data
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit(descs_train)
    # terms = vectorizer.get_feature_names()
    term_document_matrix_train = vectorizer.transform(descs_train)

    # create actual Bernoulli classifier
    classifier = BernoulliNB(alpha=1)
    classifier.fit(term_document_matrix_train, classes_train)

    term_document_matrix_test = vectorizer.transform(descs_test)
    term_document_matrix_output = vectorizer.transform(output_data)

    predicted_classes_output = classifier.predict(term_document_matrix_output)
    indices_satisfy = np.where(predicted_classes_output == allergy_name)[0]

    return np.array(indices_satisfy)


def categ_list():
    """Create a list of Cateogries from json file"""
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    cat_list = []
    for item in nutrients_data:
        food_group = item["FoodGroup"]
        if food_group not in cat_list:
            cat_list.append(food_group)
    f.close()
    return cat_list


def stop_words():
    """Defines short words in english language. Probably a better way to do this
    is to import an nltk corpus stop_words. But I was afraid it might not work
    with heroku
    """
    stop_words = []
    f = open("words", "r")
    for x in f:
        line = re.sub('[\n]', '', x)
        stop_words.append(line)
    f.close()
    # with open_resource('static/poemInput.txt') as f:
    #     for x in f:
    #         line = re.sub('[\n]', '', x)
    #         stop_words.append(line)
    return stop_words


def split_cat(cat_list):
    """ Split categories into individual groups.
    Returns a dictionary with food group and parent food group
    Example split_cat_dict[dairy] = dairy and eggs product
    """
    cat_list_1 = {}
    stop_words_1 = stop_words()
    for word in cat_list:
        if word == 'American Indian/Alaska Native Foods':
            words = ['American Indian', 'Alaska Native', 'Native']
        else:
            word_1 = word_tokenize(word)
            words = [w for w in word_1 if not w in stop_words_1]
        for food in words:
            if food != 'Products' and food != 'Foods':
                # shouldn't this be appending to a list
                cat_list_1[food] = word
    return cat_list_1


def list_nutrients():
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    final_list = []
    prem_list = []
    non_nutrients = ["ID", "FoodGroup", "ShortDescrip", "Descrip",
                     "CommonName", "MfgName", "ScientificName"]
    for key in nutrients_data[0].keys():
        if key not in non_nutrients:
            if key != "Energy_kcal":
                nut_lst = key.split("_")
                final_list.append((nut_lst[0], key, nut_lst[1]))
            else:
                final_list.append(("Calories", key, "kcal"))

    f.close()
    for nutrient in final_list:
        nut = nutrient[1].lower()
        if nut.find("usrda") == -1:
            nut = nutrient[0].lower()
            # print("Nut is:" + nut)
            if nut.find("vit") != -1:
                vit_index = nut.find("vit")
                word_1 = nutrient[0]
                word_1 = "Vitamin " + word_1[(vit_index)+3:]
                nutrient = (word_1, nutrient[1], nutrient[2])
            prem_list.append(nutrient)
    return prem_list


# rv_allergens, pos_allergens = reverse_allergen(allergy_dict), allergen_val(
#     allergy_dict)


@irsystem.route('/', methods=['GET'])
# if nutrient_tup not in output:
def search_3():
    # get list of nutrients and category name
    query_desc = request.args.get('search')
    nutr_list = request.args.getlist('nutrients')
    cat_list = request.args.getlist('cat_search')
    allergy_list = request.args.getlist('allergies')
    allergy_list=allergy_filter(allergy_list)
    
    # if anthing is blank do nothing else put nutrients into list and pass category name with it to processing _data function
    if not query_desc and not nutr_list and not cat_list and not allergy_list:
        data = []
        output_message = ''
    else:
        nutr_val = []
        for nutr in nutr_list:
            nutr_val.append(nutr)
        output_message = "Your searched for: " 
        
        if cat_list:
            category_list = category_filtering(str(cat_list))
        else:
            category_list = json.load(open('datasets/nutrients6.json',))
        if nutr_val:
            # print("Category List is: " + str(category_list))
            nutr_list = nutrients_filtering(category_list, nutr_val)
        else:
            nutr_val = []
            nutr_list = category_list
        if query_desc:
            # print("HERE 3")
            # This works only if a user provides a description list
            if len(query_desc.split()) > 1:
                desc_filt_list = descrip_filtering(query_desc, nutr_list, True)
                if len(desc_filt_list) == 0:
                    desc_filt_list = descrip_filtering(
                        query_desc, nutr_list, False)
            else:
                desc_filt_list = descrip_filtering(
                    query_desc, nutr_list, False)
            if desc_filt_list == []:
                desc_filt_list = review_filtering(query_desc, nutr_list)
                # print("DESC FILT LIST: " + str(len(desc_filt_list)))
            desc_list = rank_results(desc_filt_list, nutr_val)
        else:
            # print("HERE 2")
            desc_filt_list = nutr_list
            # rank_Results 2 do later
            desc_list = rank_results(desc_filt_list, nutr_val)
        if desc_list is None:
            desc_list = []
        # print("DESC LIST IS" + str(len(desc_list)))
        # print(desc_list)
        if desc_list != []:
            # print("HERE")
            for allergy in allergy_list:
                output_indices = bernoulli_nb(allergy, desc_list)
                desc_list = np.array(desc_list)[output_indices]
                desc_list = np.ndarray.tolist(desc_list)
            # print("ALEERGY LIST IS" + str(len(desc_list)))
        random.shuffle(desc_list)
        data = desc_list[:10]

    return render_template('botc_final_final.html', name=project_name, netid=net_id,output_message=output_message, data=data, nutr_list=list_nutrients(), cat_list=categ_list(), allergies=allergy_dict)


def review_filtering(desc, food_items):
    # f = open('datasets/nutrients3.json',)
    # food_items = json.load(f)
    # food_items = food_items[:5]
    final_list = []
    for item in food_items:
        for review in item["review"]:
            for actual_review in review:
                list_of_words = actual_review.split()
                if intersect_old(desc, list_of_words):
                    if item not in final_list:
                        final_list.append(item)
    return final_list
    # print(review)
    f.close()


def category_filtering(query_categories):
    """Filter query categories to include only relevant categories
    """
    f = open('datasets/nutrients6.json',)
    nutrients_data = json.load(f)
    category_list = categ_list()
    split_cat_dict = split_cat(category_list)
    # print(split_cat_dict)
    # separate between input to get list
    q_cat_list = query_categories.split(",")
    # Split query list into individual categories just like original categories in json
    q_split_cat = split_cat(q_cat_list)
    split_cat_keys = split_cat_dict.keys()
    output = []

    # use boolean search to match specific food groups first
    for food_group in q_split_cat:
        if food_group in split_cat_keys:
            food_item = split_cat_dict[food_group]
            if food_item not in output:
                output.append(food_item)

    # if ouput is zero user might have mispelled a food group, so every food group in the list
    # use edit distance to retrieve right nutrient. We should probably set a threshold for edit distance because certain outputs don't
    # make sense
    # else continue
    if len(output) == 0:
        for fgroup in q_split_cat:
            valid_category = edit_distance_search(fgroup, split_cat_keys)
            f_group = split_cat_dict[valid_category[1]]
            if f_group not in output:
                output.append(f_group)

    # use final cats to find food items
    food_output = []
    for food in nutrients_data:
        fgroup = food['FoodGroup']
        if fgroup in output:
            # food id is put into a list: food_output
            # food_item = {"FoodGroup": fgroup,
            #              "ShortDescrip": food['ShortDescrip'], "Descrip":  food['Descrip'], "MfgName":  food['MfgName']}
            food_output.append(food)
    return food_output


def nutrients_filtering(cat_output, query_nutrients):
    """Filter nutrients and rank results based on food_groups that meet the minimum daily requirement
    for a 2000 calorie diet
    """
    # Calorie level based on a 2000 Calorie diet
    calorie_level = {'Protein_g': 34, 'Fat_g': 44, 'Carb_g': 130, 'Sugar_g': 25,
                     'Fiber_g': 28, 'VitA_mcg': 700, 'VitB6_mg': 1.3, 'VitB12_mcg': 2.4,
                     'VitC_mg': 75, 'VitE_mg': 15, 'Folate_mcg': 400, 'Niacin_mg': 14,
                     'Riboflavin_mg': 1.1, 'Thiamin_mg': 1.1, 'Calcium_mg': 1000,
                     'Copper_mcg': 900, 'Iron_mg': 18, 'Magnesium_mg': 310,
                     'Manganese_mg': 1.8, 'Phosphorus_mg': 700, 'Selenium_mcg': 55, 'Zinc_mg': 8, "Energy_kcal": 1000}

    nutr_out = []
    nutr_list = []
    for x in list_nutrients():
        nutr_list.append(x[1])

    # i = 0
    nutrition_list = []
    for nutrient in query_nutrients:
        nutrient = edit_distance_search(nutrient, nutr_list)
        if nutrient not in nutrition_list:
            nutrition_list.append(nutrient)
    for food_item in cat_output:
        for nutrient in nutrition_list:
            # print(nutrient)
            if nutrient[1] in calorie_level:
                cal_val = calorie_level[nutrient[1]]*0.25
                # print("Cal val" + str(cal_val))
            else:
                cal_val = 0
            if float(food_item[nutrient[1]]) > cal_val:
                # isnt this the same as before??
                # i += 1
                nutr_out.append(food_item)
                # if i >= 4000:
                #     return nutr_out
                break
    return nutr_out


def descrip_filtering(query_desc, nutr_out, advanced):
    """Return a list of food items based on the nutrient input
    Return type is a list of dictionaries
    """
    stop_words_1 = stop_words()
    quer_desc = query_desc.lower()
    quer_desc = str(set(quer_desc.split()))
    descrip_list = []
    # Stemming
    ps = PorterStemmer()
    word_tokens = word_tokenize(query_desc)
    word_tokens_1 = [w for w in word_tokens if not w in stop_words_1]
    stem_set = set([ps.stem(word) for word in word_tokens_1])
    stem_length = len(stem_set)
   # Find the intersection between the query description and the food description
    # and if it's greater than 0, then there's a match.
    for food_item in nutr_out:
        # Stem the descriptions in json file
        longd = word_tokenize(food_item['Descrip'].lower())
        set_longd = set([ps.stem(descp) for descp in longd])
        if advanced:
            if len(stem_set.intersection(set_longd)) == stem_length:
                if food_item not in descrip_list:
                    descrip_list.append(food_item)
                    continue
        else:
            if len(stem_set.intersection(set_longd)) > 0:
                if food_item not in descrip_list:
                    descrip_list.append(food_item)
                    continue

    return descrip_list
    # long_descrip=set(word_tokenize(food_item['Descrip']))
    #   food_item = {"FoodGroup": fgroup,
    #              "ShortDescrip": food['ShortDescrip'], "Descrip":  food['Descrip'], "MfgName":  food['MfgName']}


def curr_insertion_function(message, j):
    return 1


def curr_deletion_function(query, i):
    return 1


def curr_substitution_function(query, message, i, j):
    if query[i-1] == message[j-1]:
        return 0
    else:
        return 1


def edit_matrix(query, message):

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i-1, 0] + curr_deletion_function(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j-1] + curr_insertion_function(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i-1, j] + curr_deletion_function(query, i),
                chart[i, j-1] + curr_insertion_function(message, j),
                chart[i-1, j-1] +
                curr_substitution_function(query, message, i, j)
            )
    return chart


def edit_distance(query, message):
    query = query.lower()
    message = message.lower()

    chart = edit_matrix(query, message)

    return chart[len(query), len(message)]


def edit_distance_search(query, msgs):
    output = []
    for list1 in msgs:
        score = edit_distance(query, list1)
        output.append((score, list1))
    fin = sorted(output, key=lambda x: x[0])
    return fin[0]

def allergy_filter(allergy_list):
    output=[]
    for alley in allergy_list:
        val=edit_distance_search(alley, allergy_dict.keys())
        output.append(val)
    return output



def rank_results(descript_list, query_nutrients):
    """ This function ranks the results of the description filter based on the
    nutrient query. Results with higher input nutrient type will rank higher than results
    with lower input nutrient type.
    For a query like Calcium and Protein and a list of viable descriptions
    This function will return a dictionary like:

    {protein: [{Nutrient Info1},....
    Calcium: [{Nutrient Info1}, ...}]}
    """
    if query_nutrients:
        output = {}
        nut_score_dict = {}
        nutr_list = []
        final_ranks = []
        for x in list_nutrients():
            nutr_list.append(x[1])
        # print("NUTRIENT LIST IS" + str(nutr_list))
        for nutrient in query_nutrients:
            nutrient_1 = edit_distance_search(nutrient, nutr_list)
            nut = nutrient_1[1]
            for item in descript_list:
                if nut not in nut_score_dict:
                    nut_score = float(item[nut])
                    nut_score_dict[nut] = [(item, nut_score)]
                elif nut in nut_score_dict:
                    nut_score = float(item[nut])
                    nut_score_dict[nut].append((item, nut_score))

        # print("NUTRIENT SCORE DICT: " + str(nut_score_dict))
        for nutrient in nut_score_dict.keys():
            nutrient_list = nut_score_dict[nutrient]
            fin = sorted(nutrient_list, key=lambda x: x[1], reverse=True)
            output[nutrient] = fin

        for nut_data in output.keys():
            rank_set = list(output[nut_data])
            for i in range(len(rank_set)):
                final_ranks.append(rank_set[i][0])
            return final_ranks
    else:
        return descript_list


def rank_results2(query_nutrients):
    """Use this function if the user does not provide a description
    """
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    ranks = rank_results(nutrients_data, query_nutrients)
    return ranks
