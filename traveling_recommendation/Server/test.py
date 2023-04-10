# import numpy as np
# from flask import Flask, render_template, request
# from flaskext.mysql import MySQL
# import face_model, recommend_model, os
#
# UPLOAD_FOLDER = 'static/image'
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# file_list = face_model.save_files(request, app.config['UPLOAD_FOLDER'])
# face_feeling = face_model.get_img_path(file_list)
#
# print(face_feeling)
#
#
# if __name__ == "__main__":
#     app.secret_key = 'super secret key'
#     app.config['SESSION_TYPE'] = 'filesystem'
#     app.run(host='0.0.0.0', port='5050')
