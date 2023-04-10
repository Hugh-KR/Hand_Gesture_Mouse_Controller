import numpy as np
from flask import Flask, render_template, request, jsonify
from flaskext.mysql import MySQL
import recommend_model, os
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse

# CORS(Cross Origin Resource Sharing) : 동일 출처(같은 호스트네임)가 아니더라도 정상적으로 사용 가능하도록 도와주는 방법입니다.
# 다른 도메인이나 로컬 환경에서 자바스크립트로 api 등을 호출하는 경우 브라우저에서 동일 출처 위반의 에러가 나타날 수 있습니다

from flask_cors import CORS

UPLOAD_FOLDER = 'static/image'
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_USER'] = 'abcd'
app.config['MYSQL_DATABASE_PASSWORD'] = '12345678'
app.config['MYSQL_DATABASE_DB'] = 'mydb'

mysql = MySQL()
mysql.init_app(app)

quests = [2,    1,	2,	3,	4,	4,	1,	3,	3,	5,	3,	3,	2,	3,	3,	5,	3,	4]




print(quests)

@app.route("/plus", methods=["GET"])
def index():

    place_visit_prop = []
    placelist = []
    image_list = []
    tag_list = []
    url_list = []


    place_table = get_table()

    for place in place_table:
        placelist.append(place['place'])
        tag_list.append(place['tag'])
        url_list.append(place['URL'])

        input_data = quests \
                    + [float(place['theme_edu']), float(place['theme_act']), float(place['theme_nat'])] \
                    + [1]
        prop = recommend_model.recommend_place(input_data)
        place_visit_prop.append(prop)

    place_visit_prop = np.array(place_visit_prop)
    print(placelist)
    place_visit_top6 = np.flip(np.argpartition(place_visit_prop, -6)[-6:])
    print(list(place_visit_top6))
    tag_list = [tag_list[x].split(" ") for x in list(place_visit_top6)]
    url_list = [url_list[y] for y in list(place_visit_top6)]
    recommend_place = [placelist[j] for j in list(place_visit_top6)]
    print(recommend_place)

    for place in recommend_place:
        image_list.append('image/place/' + place + '.jpg')
    print(tag_list)

    return render_template('index.html', image_list=image_list, place_list=recommend_place, tag_list=tag_list, url_list=url_list)


def get_table():
    cursor = mysql.get_db().cursor()
    cursor.execute('select * from travel_place_1')

    column_names = [x[0] for x in cursor.description]
    results = cursor.fetchall()
    place_list = []

    for result in results:
        place_list.append(dict(zip(column_names, result)))
    print(place_list)
    return place_list


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port='5050')
