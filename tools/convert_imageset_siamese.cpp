// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "opencv2/opencv.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

void CVMatPairToDatum(const cv::Mat& cv_img1, const cv::Mat& cv_img2, Datum* datum) 
{
	CHECK(cv_img1.depth() == CV_8U) << "Image data type must be unsigned byte";
	CHECK(cv_img2.depth() == CV_8U) << "Image data type must be unsigned byte";
	CHECK(cv_img1.depth() == cv_img2.depth()) << "Image data type must be unsigned byte";

	datum->set_channels(cv_img1.channels() + cv_img2.channels());
	datum->set_height(cv_img1.rows);
	datum->set_width(cv_img1.cols);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	int c;
	std::string buffer(datum_size, ' ');
	for (int h = 0; h < datum_height; ++h) {
		const uchar* ptr = cv_img1.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; ++w) {
			for (c = 0; c < cv_img1.channels(); ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);
			}
		}
	}
	for (int h = 0; h < datum_height; ++h) {
		const uchar* ptr = cv_img2.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; ++w) {
			for (c = cv_img1.channels(); c < datum_channels; ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);
			}
		}
	}
	datum->set_data(buffer);
}

bool ReadImagePairToDatum(const string& filename1, const string& filename2, 
	const int label1, const int label2,
	const int height, const int width, const bool is_color,
	const std::string & encoding, Datum* datum) {
	cv::Mat cv_img1 = ReadImageToCVMat(filename1, height, width, is_color);
	cv::Mat cv_img2 = ReadImageToCVMat(filename2, height, width, is_color);
	if (cv_img1.data && cv_img2.data) 
	{
		CVMatPairToDatum(cv_img1, cv_img2, datum);
		if (label1 == label2) {
			datum->set_label(1);
		}
		else {
			datum->set_label(0);
		}
		return true;
	}
	else {
		return false;
	}
}

// equProb 挑出来两个的标签相等的概率
int pickPairWithProb(int &pn1, int &pn2, 
	std::vector<std::pair<std::string, int> > list, int total, float equProb)
{
	int n = (int)(equProb * 10 + 0.5f);
	int prob = caffe::caffe_rng_rand() % 10;  
	pn1 = caffe::caffe_rng_rand() % total;
	int label = list[pn1].second;
	if (prob < n) // 相等
	{
		int iter = 5000;
		while (iter--)
		{
			pn2 = caffe::caffe_rng_rand() % total;
			if (list[pn2].second == label && pn2 != pn1)
				break;
		}
		if (iter <= 0) // 查找失败
			pn2 = pn1;
	}
	else
	{
		while (1)
		{
			pn2 = caffe::caffe_rng_rand() % total;
			if (list[pn2].second != label)
				break;
		}
	}
	return 1;
}

int main(int argc, char** argv) {

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  caffe:GlobalInit(&argc, &argv);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  int nimg;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  nimg = lines.size();
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
	int pn1, pn2;
	//pn1 = caffe::caffe_rng_rand() % nimg;  // pick a random  pair
	//pn2 = caffe::caffe_rng_rand() % nimg;
	pickPairWithProb(pn1, pn2, lines, nimg, 0.5f);
    status = ReadImagePairToDatum(root_folder + lines[pn1].first,
		root_folder + lines[pn2].first, lines[pn1].second, lines[pn2].second, 
		resize_height, resize_width, is_color,
        enc, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
	int length = sprintf_s(key_cstr, kMaxKeyLength, "%08d", line_id);
        //lines[line_id].first.c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
