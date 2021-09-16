# -*- coding: utf-8 -*-
from flask import Flask, jsonify, abort, make_response
# import json

api = Flask(__name__)

@api.route('/info/<string:userId>', methods=['GET'])
def get_info(userId):
    print(userId)

    result = {
        "result":True,
        "data":{
            "userId":userId
            }
        }

    return make_response(jsonify(result))
    # Unicodeにしたくない場合は↓
    # return make_response(json.dumps(result, ensure_ascii=False))

@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    api.run(debug=True)
