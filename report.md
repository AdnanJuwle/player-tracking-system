# Player Tracking and Re-Identification System — Report

## Approach and Methodology

The project aims to track football players consistently within a single camera feed and across multiple camera feeds.

- For **single-feed tracking**, I used YOLOv11 for player detection and DeepSORT for tracking players across frames.
- For **cross-camera re-identification**, the approach involves extracting appearance features using a ResNet50-based embedding model and matching player identities between two synchronized camera feeds using cosine similarity and the Hungarian matching algorithm.

The methodology builds upon my prior experience with YOLOv5, initially applying it but switching to YOLOv11 for better detection performance and stability.

## Techniques Tried and Their Outcomes

| Technique                    | Outcome                                         | Notes                                  |
|-----------------------------|------------------------------------------------|----------------------------------------|
| YOLOv5 + Simple Tracker      | Adequate for short-term single-feed tracking  | IDs reset when players left and re-entered |
| YOLOv11 + DeepSORT           | Stable and consistent single-feed tracking    | Successfully maintained IDs across video duration |
| Cross-Camera ReID (embedding + Hungarian) | Partial functionality                         | Cross-camera matching unreliable due to appearance variance and synchronization issues |

## Challenges Encountered

- **Cross-camera Re-identification not fully functional:** Appearance differences, lighting variations, and lack of precise synchronization between feeds impacted matching accuracy.
- **CUDA/GPU Environment issues:** GPU was not accessible in the development environment, causing fallback to CPU and slower performance.
- **Feature extraction speed:** Extracting embeddings for each detected player frame-by-frame was computationally expensive.
- **Occlusions and missed detections:** Occasionally caused ID switches or loss of tracking in single-feed mode.

## Incomplete Work and Future Directions

- The **cross-camera re-identification** module requires further development.
- Given more time and resources, I would:
  - Implement temporal alignment of multi-camera feeds to improve synchronization.
  - Train a domain-specific ReID model tailored for football players to reduce embedding mismatches.
  - Optimize feature extraction for real-time inference.
  - Incorporate auxiliary features such as jersey numbers or color histograms to aid matching.
  - Enhance tracking robustness with motion models like Kalman filters or LSTMs.
  - Perform quantitative evaluation using multi-object tracking benchmarks.

---

This project provided valuable practical experience integrating detection, tracking, and re-identification modules into a coherent system for player tracking in sports videos.

