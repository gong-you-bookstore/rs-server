from flask import Flask, request, jsonify
from flask_cors import CORS
from recommend import find_sim_books
import json

app = Flask(__name__)

CORS(app)


@app.route('/')
def root():
    return 'Server is running'


# GET: 책 한권의 isbn을 파라미터로 보내면 유사도 높은 책 20개 반횐
# 책 상세페이지 에서 사용할 것
# 예시: http://127.0.0.1:5000/recommend/one?isbn=9791156759270
@app.route('/recommend/one', methods=['GET'])
def recommend_one():
    params = request.args.get('isbn')
    result = find_sim_books([int(params)])

    return jsonify({
        'message': 'success',
        'data': result
    })

# POST: 여러 책의 isbn 값을 넘기면 책 추천 받음
# 유저가 등록한 책들의 리스트를 인풋으로 받을 것
# 예시: http://127.0.0.1:5000/recommend/multi


@app.route('/recommend/multi', methods=['GET', 'POST'])
def recommend_multi():
    body = request.get_json()
    book_list = list(map(int, body['data']))
    result = find_sim_books(book_list)

    return jsonify({
        'message': 'success',
        'data': result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
