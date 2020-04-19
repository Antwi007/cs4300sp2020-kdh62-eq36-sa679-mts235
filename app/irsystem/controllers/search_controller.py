from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import json

project_name = "Project Re-search"
net_id = "Nana Antwi: nka32, Max Stallop: mls235, Edwin Quaye: eq36, Stephen Adusei Owusu: sa679, Kenneth Harlley: kdh62"

# f = open('nutrients.json',)
# nutrients_data = json.load(f)


@irsystem.route('/', methods=['GET'])
def search():
    # get list of nutrients and category name
    query = request.args.get('search')
    nutr_list = request.args.getlist('nutrients')
    cat_list=request.args.get('cat_search')
    final=category_filtering(str(cat_list))

    # if anthing is blank do nothing else put nutrients into list and pass category name with it to processing _data function
    if not query or not nutr_list:
        data = []
        output_message = ''
    else:
        query_val = []
        for nutr in nutr_list:
            query_val.append(nutr)
        output_message = "Your search: " + query
        #final=category_filtering(cat_list)
        category_name = query
        output = processing_data(query_val, category_name)
        data = output

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data,fin=final)


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
    #separate between input to get list
    cat_list=query_categories.split(",")
    #use edit distance to find actual cats from input
    output=[]
    for nutrient in cat_list:
        valid_nutrient=edit_distance_search(nutrient,['Vegetables and Vegetable Products', 'Soups, Sauces, and Gravies', 'Baby Foods', 'Fruits and Fruit Juices', 'American Indian/Alaska Native Foods', 'Finfish and Shellfish Products', 'Sweets', 'Beverages', 'Snacks', 'Nut and Seed Products', 'Beef Products', 'Fats and Oils', 'Restaurant Foods', 'Lamb, Veal, and Game Products', 'Dairy and Egg Products', 'Legumes and Legume Products', 'Poultry Products', 'Fast Foods', 'Meals, Entrees, and Side Dishes', 'Spices and Herbs', 'Baked Products', 'Sausages and Luncheon Meats', 'Breakfast Cereals', 'Cereal Grains and Pasta', 'Pork Products']
)
        output.append(valid_nutrient)

    return output






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
        chart[i,0] = chart[i-1, 0] + curr_deletion_function(query, i) 
    for j in range(1, n): 
        chart[0,j] = chart[0, j-1] + curr_insertion_function(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i-1, j] + curr_deletion_function(query, i),
                chart[i, j-1] + curr_insertion_function(message, j),
                chart[i-1, j-1] + curr_substitution_function(query, message, i, j)
            )
    return chart

def edit_distance(query, message):        
    query = query.lower()
    message = message.lower()
    
    chart=edit_matrix(query,message)
    
    return chart[len(query),len(message)]

def edit_distance_search(query, msgs):
    output=[]
    for list1 in msgs:
        score=edit_distance(query,list1)
        output.append((score,list1))
    fin=sorted(output, key=lambda x: x[0])
    return fin[0]