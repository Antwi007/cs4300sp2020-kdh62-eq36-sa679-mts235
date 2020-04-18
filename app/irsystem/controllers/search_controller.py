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
    # if anthing is blank do nothing else put nutrients into list and pass category name with it to processing _data function
    if not query or not nutr_list:
        data = []
        output_message = ''
    else:
        query_val = []
        for nutr in nutr_list:
            query_val.append(nutr)
        output_message = "Your search: " + query
        category_name = query
        output = processing_data(query_val, category_name)
        data = output

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


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
