# coding=utf-8
import os
import make_evaluate
import PltShow
from flask import Flask
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = './im_test/'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def classify():
    if request.files.get('file'):
        result = []
        files = request.files.getlist('file')
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename = secure_filename(file.filename)
            # results = make_evaluate.evaluate_one_image(filename)
            results = make_evaluate.main(filename)
            result.append(results)
        PltShow.show_result(result)

        return render_template('index.html', **locals())
    else:
        return index()


if __name__ == '__main__':
    app.run(debug=True)