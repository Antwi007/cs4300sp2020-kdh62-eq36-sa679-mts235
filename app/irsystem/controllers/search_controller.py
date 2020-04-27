from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

project_name = "Project Re-search"
net_id = "Nana Antwi: nka32, Max Stallop: mls235, Edwin Quaye: eq36, Stephen Adusei Owusu: sa679, Kenneth Harlley: kdh62"

# f = open('nutrients.json',)
# nutrients_data = json.load(f)


def categ_list():
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    cat_list = []
    for item in nutrients_data:
        food_group = item["FoodGroup"]
        if food_group not in cat_list:
            cat_list.append(food_group)
    f.close()
    return cat_list


def list_nutrients():
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    final_list = []
    non_nutrients = ["ID", "FoodGroup", "ShortDescrip", "Descrip",
                     "CommonName", "MfgName", "ScientificName", "Energy_kcal"]
    for key in nutrients_data[0].keys():
        if key not in non_nutrients:
            nut_lst = key.split("_")
            final_list.append((nut_lst[0], key))
    f.close()
    return final_list


@irsystem.route('/about', methods=['GET'])
def search():
    # get list of nutrients and category name
    query_desc = request.args.get('search')
    nutr_list = request.args.getlist('nutrients')
    cat_list = request.args.get('cat_search')
    p_list = request.args.getlist('selret')
    print("p_list IS" + str(p_list))
    # final = category_filtering(str(cat_list))

    # if anthing is blank do nothing else put nutrients into list and pass category name with it to processing _data function
    if not query_desc and not nutr_list and not cat_list:
        print("HERE 1")
        data = []
        output_message = ''
    else:
        print("HERE 2")
        nutr_val = []
        for nutr in nutr_list:
            nutr_val.append(nutr)
        output_message = "Your search: " + query_desc
        # final = category_filtering(cat_list)
        # category_name = query
        if cat_list:
            category_list = category_filtering(str(cat_list))
        else:
            category_list = json.load(open('nutrients.json',))
        if nutr_val:
            # print("Category List is: " + str(category_list))
            nutr_list = nutrients_filtering(category_list, nutr_val)
        else:
            nutr_list = category_list
        if query_desc:
            # print("Nutrient List is: " + str(nutr_list))
            desc_list = descrip_filtering(query_desc, nutr_list)
        else:
            desc_list = nutr_list
        # print("Description List is: " + str(desc_list))
        # output
        # for desc in desc_list:

        # output = processing_data(query_val, category_name)
        # if len(desc_list) <= 20:
        data = desc_list[:6]
        # else:
        #     data = desc_list[-1]+desc_list[1] + \
        #         desc_list[len(desc_list)//2]+desc_list[len(desc_list) //
        #                                                4] + desc_list[len(desc_list)//8]

    return render_template('boltc.html', name=project_name, netid=net_id, output_message=output_message, data=data, nutr_list=list_nutrients(), cat_list=categ_list())


def processing_data(query_nutrients, category_name):
    output = []
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    # Loop through all the different food itwms storing their name and category
    for i in range(len(nutrients_data)):
        name = nutrients_data[i]["ShortDescrip"]
        category = nutrients_data[i]["FoodGroup"]
        # for every nutrient requested in the query, compare it with the current food item.
        # If the food item has the correct nutrient and also is in the category name add it to the output
        # and remove the nutrient from the list of requested nutrients since it has been added to the grocery list
        for nutrient in query_nutrients:
            if float(nutrients_data[i][nutrient]) > 0 and category == category_name:
                nutrient_tup = (name, category)
                # if the nutrient is already in the ouptu don't add
                if nutrient_tup not in output:
                    output.append(nutrient_tup)
                query_nutrients.remove(nutrient)
        if query_nutrients == []:
            break
    f.close()
    return output


def category_filtering(query_categories):
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    # separate between input to get list
    cat_list = query_categories.split(",")
    # use edit distance to find actual cats from input
    output = []
    for nutrient in cat_list:
        valid_nutrient = edit_distance_search(nutrient, ['Vegetables and Vegetable Products', 'Soups, Sauces, and Gravies', 'Baby Foods', 'Fruits and Fruit Juices', 'American Indian/Alaska Native Foods', 'Finfish and Shellfish Products', 'Sweets', 'Beverages', 'Snacks', 'Nut and Seed Products', 'Beef Products', 'Fats and Oils', 'Restaurant Foods', 'Lamb, Veal, and Game Products', 'Dairy and Egg Products', 'Legumes and Legume Products', 'Poultry Products', 'Fast Foods', 'Meals, Entrees, and Side Dishes', 'Spices and Herbs', 'Baked Products', 'Sausages and Luncheon Meats', 'Breakfast Cereals', 'Cereal Grains and Pasta', 'Pork Products']
                                              )
        output.append(valid_nutrient[1])

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
    nutr_out = []
    nutr_list = []
    for x in list_nutrients():
        nutr_list.append(x[1])
    for food_item in cat_output:
        for nutrient in query_nutrients:
            # print("THIS IS NUTRIENT" + str(nutrient))
            nutrient = edit_distance_search(nutrient, nutr_list)
            # print("THIS IS NUTRIENT" + str(nutrient))
            if float(food_item[nutrient[1]]) > 0:
                nutr_out.append(food_item)
                break
    return nutr_out
    # nutrient_tup = (name, category)
    # # if the nutrient is already in the ouptu don't add
    # if nutrient_tup not in output:
    #     output.append(nutrient_tup)
    # query_nutrients.remove(nutrient)


def descrip_filtering(query_desc, nutr_out):
    quer_desc = query_desc.lower()
    descrip_list = []
    stem_set = set(quer_desc.split())
    # ps = PorterStemmer()
    # word_tokens = word_tokenize(query_desc)
    # stem_set = set(map(ps.stem, word_tokens))
    # q_tokens = set(word_tokenize(query_desc))
    for food_item in nutr_out:
        # shortd = word_tokenize(food_item['ShortDescrip'].lower())
        # set_shortd = set(map(ps.stem, shortd))
        # add support for st
        shortd = food_item['ShortDescrip'].lower()
        set_shortd = set(shortd.split())

        # longd = word_tokenize(food_item['Descrip'].lower())
        # set_longd = set(map(ps.stem, shortd))
        longd = food_item['ShortDescrip'].lower()
        set_longd = set(longd.split())
        # if
        if len(set_longd.intersection(stem_set)) != 0 or len(set_longd.intersection(stem_set)) != 0:
            descrip_list.append(food_item)
            continue
        # if inside somewhere
        for word in query_desc:
            if word.find(longd) != -1:
                descrip_list.append(food_item)
                break
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
