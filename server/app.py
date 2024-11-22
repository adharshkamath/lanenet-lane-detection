from flask import Flask, request, jsonify
from lanenet import LaneNetServer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


app = Flask(__name__)
lnserver = LaneNetServer(weights_path='weights/tusimple_lanenet.ckpt')


@app.route('/inference', methods=['POST'])
def inference():
    image = request.files['image']
    output = lnserver.run_inference(image=image, with_lane_fit=True)
    return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)
