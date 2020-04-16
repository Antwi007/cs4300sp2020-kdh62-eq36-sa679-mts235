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
    query = request.args.get('search')
    nutr_list = request.args.getlist('nutrients')
    # thiamin_c = request.args.get("thiamin_c")
    # print("THIAMIN VALUE IS:" + thiamin_c)
    # print("VITAMIN C VALUE IS:" + str(vitc_val))
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
        # print(nutrients_data[2])
        # data = range(5)

    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


def processing_data(query_nutrients, category_name):
    output = []
    f = open('nutrients.json',)
    nutrients_data = json.load(f)
    for i in range(len(nutrients_data)):
        name = nutrients_data[i]["ShortDescrip"]
        category = nutrients_data[i]["FoodGroup"]
        for nutrient in query_nutrients:
            if float(nutrients_data[i][nutrient]) > 0 and category == category_name:
                nutrient_tup = (name, category)
                if nutrient_tup not in output:
                    output.append(nutrient_tup)
                query_nutrients.remove(nutrient)
        if query_nutrients == []:
            break
    f.close()
    return output
