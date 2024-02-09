#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

std::vector<cv::String> get_outputs_names(const cv::dnn::Net& net);
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<cv::String>& class_labels);

int main() {
    cv::String model_configuration = "../data/yolov3.cfg";
    cv::String model_weights = "../data/yolov3.weights";
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(model_configuration, model_weights);
    std::vector<cv::String> class_labels;
    cv::String classes_file = "../data/coco.names";
    std::ifstream ifs(classes_file.c_str());
    std::string line;
    while (getline(ifs, line)) {
        class_labels.push_back(line);
    }
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open webcam" << std::endl;
        return -1;
    }
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    while (cv::waitKey(1) < 0) {
        cv::Mat frame;
        cap >> frame;
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, get_outputs_names(net));
        postprocess(frame, outs, class_labels);
        imshow("Object detection", frame);
    }
}

std::vector<cv::String> get_outputs_names(const cv::dnn::Net& net) {
    std::vector<cv::String> names;
    if (names.empty()) {
        std::vector<int> out_layers = net.getUnconnectedOutLayers();
        std::vector<cv::String> layers_names = net.getLayerNames();
        names.resize(out_layers.size());
        for (size_t i = 0; i < out_layers.size(); ++i) {
            names[i] = layers_names[out_layers[i] - 1];
        }
    }
    return names;
}

// draw bounding box around detected objects and label them
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<cv::String>& class_labels) {
    int frame_width = frame.cols;
    int frame_height = frame.rows;
    float conf_threshold = 0.5;
    float nms_threshold = 0.4;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point class_id_point;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > conf_threshold) {
                int width = (int)(data[2] * frame_width);
                int height = (int)(data[3] * frame_height);
                int center_x = (int)(data[0] * frame_width);
                int center_y = (int)(data[1] * frame_height);
                int left = center_x - width / 2;
                int top = center_y - height / 2;
                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    // non-maximum suppression for eliminating overlapping bounding boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        float confidence = confidences[idx];
        rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        cv::String label = class_labels[class_id] + ": " + cv::format("%.2f", confidence);
        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int top = std::max(box.y - 15, labelSize.height);
        putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}
