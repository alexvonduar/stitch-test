//
//  Blending.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "Blending.h"

Rect resultRoi(const std::vector<Point> &corners, const std::vector<Size> &sizes)
{
	CV_Assert(sizes.size() == corners.size());
	Point tl(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
	Point br(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
	for (size_t i = 0; i < corners.size(); ++i)
	{
		tl.x = std::min(tl.x, corners[i].x);
		tl.y = std::min(tl.y, corners[i].y);
		br.x = std::max(br.x, corners[i].x + sizes[i].width);
		br.y = std::max(br.y, corners[i].y + sizes[i].height);
	}
	return Rect(tl, br);
}

Mat getMatOfLinearBlendWeight(const Mat & image) {
    Mat result(image.size(), CV_32FC1, Scalar::all(0));
    for(int y = 0; y < result.rows; ++y) {
        int w_y = min(y + 1, result.rows - y);
        for(int x = 0; x < result.cols; ++x) {
            result.at<float>(y, x) = min(x + 1, result.cols - x) * w_y;
        }
    }
    return result;
}

vector<Mat> getMatsLinearBlendWeight(const vector<Mat> & images) {
    vector<Mat> result;
    result.reserve(images.size());
    for(int i = 0; i < images.size(); ++i) {
        result.emplace_back(getMatOfLinearBlendWeight(images[i]));
    }
    return result;
}

Mat Blending(const vector<Mat> & images,
             const vector<Point> & corners,
			 const vector<Size> sizes,
             const Size2 target_size,
             const vector<Mat> & weight_mask,
             const bool ignore_weight_mask) {
	
	int num_images = corners.size();
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
   
	for (int i = 0; i < num_images; ++i)
	{
		masks_warped[i] = weight_mask[i].getUMat(ACCESS_READ);
		images_warped[i] = images[i].getUMat(ACCESS_READ);
		cvtColor(images_warped[i], images_warped[i], CV_BGRA2BGR);
	}

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = makePtr<detail::NoSeamFinder>();
	else if (seam_find_type == "voronoi")
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
	else if (seam_find_type == "gc_color")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "Using 'gc_colorgrad' instead'\n";
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);//return 1;
	}

    
	seam_finder->find(images_warped, corners, masks_warped);

	//images_warped.clear();

	Mat dilated_mask, seam_mask, mask, mask_warped, img_warped, img_warped_s;
	Ptr<Blender> blender;

    vector<Rect2> rects;
    rects.reserve(corners.size());
    for(int img_idx = 0; img_idx < corners.size(); ++img_idx) {
		img_warped.create(images[img_idx].size(), CV_8U);
		cvtColor(img_warped, img_warped, CV_BGRA2BGR);
		mask_warped = masks_warped[img_idx].getMat(ACCESS_READ);

		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		
		dilate(mask_warped, dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;
        
		if (!blender)
		{
			blender = Blender::createDefault(blend_type, try_cuda);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				fb->setSharpness(1.f / blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			blender->prepare(corners, sizes);
		}
		else 
		{
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}

    }
   
	Mat result, result_mask;
	blender->blend(result, result_mask);
	result.convertTo(result, CV_8U);
	cvtColor(result, result, CV_BGR2BGRA);
   
    return result;
}

Mat Blending(const vector<Mat> & images,
	const vector<Point2> & origins,
	const Size2 target_size,
	const vector<Mat> & weight_mask,
	const bool ignore_weight_mask) {

	Mat result = Mat::zeros(round(target_size.height), round(target_size.width), CV_8UC4);

	vector<Rect2> rects;
	rects.reserve(origins.size());
	for (int i = 0; i < origins.size(); ++i) {
		rects.emplace_back(origins[i], images[i].size());
	}
	for (int y = 0; y < result.rows; ++y) {
		for (int x = 0; x < result.cols; ++x) {
			Point2i p(x, y);
			Vec3f pixel_sum(0, 0, 0);
			float weight_sum = 0.f;
			for (int i = 0; i < rects.size(); ++i) {
				Point2i pv(round(x - origins[i].x), round(y - origins[i].y));
				if (pv.x >= 0 && pv.x < images[i].cols &&
					pv.y >= 0 && pv.y < images[i].rows) {
					Vec4b v = images[i].at<Vec4b>(pv);
					Vec3f value = Vec3f(v[0], v[1], v[2]);
					if (ignore_weight_mask) {
						if (v[3] > 127) {
							pixel_sum += value;
							weight_sum += 1.f;
						}
					}
					else {
						float weight = weight_mask[i].at<float>(pv);
						pixel_sum += weight * value;
						weight_sum += weight;
					}
				}
			}
			if (weight_sum) {
				pixel_sum /= weight_sum;
				result.at<Vec4b>(p) = Vec4b(round(pixel_sum[0]), round(pixel_sum[1]), round(pixel_sum[2]), 255);
			}
		}
	}
	return result;
}