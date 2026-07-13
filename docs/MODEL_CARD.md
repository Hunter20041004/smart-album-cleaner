# Smart Album Cleaner Classifier Model Card

This card describes the repository's MobileNetV3 Large binary classifier and
the evidence available in the tracked evaluation reports. It does not extend
the evidence beyond those reports.

## Intended Use

The classifier assigns an extracted face to one of two subjective album-cleaning
labels, `Good` or `Bad`, using a reported `P(Bad)` decision threshold of 0.440.
Its intended role is to help a person review possible cleanup candidates. It is
not identity recognition, face verification, emotion diagnosis, or a basis for
decisions about a person.

## Data

The tracked report identifies a test set of 193 examples: 99 labelled `Bad` and
94 labelled `Good`. The repository does not contain a tracked measured report
that establishes the dataset's origin, collection process, contributor consent,
licence, demographic composition, duplicate handling, or label-review process.
Those facts are unknown.

The tracked reports also do not state the training or validation sample counts,
how samples were split, or whether the reported test set is independent by
person, event, album, or source. Split independence therefore cannot be
confirmed from the available evidence.

## Evaluation

`reports/classification_report.txt` records a single evaluation of
`models/mobilenet_face.pth`, architecture `mobilenet_v3_large`, at a `P(Bad)`
threshold of 0.440. On the reported 193-example test set, accuracy was **75.1%**.
This number applies only to that binary `Good`/`Bad` evaluation and that test
set; it is not evidence of identity, demographic, real-world album, or
out-of-distribution performance.

No variance across repeated runs, confidence intervals, external benchmark,
device benchmark, or subgroup evaluation is present in the tracked measured
reports.

## Class-level Results

The following values are copied from `reports/classification_report.txt` and
are rounded there to three decimal places:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `Bad` | 0.726 | 0.828 | 0.774 | 99 |
| `Good` | 0.787 | 0.670 | 0.724 | 94 |

The lower `Good` recall (0.670) means the reported evaluation missed more
`Good` examples than its single headline accuracy communicates. Class labels
are subjective judgments about album-cleaning quality, not objective facts
about a face or person.

## Limitations and Bias

- Unknown data provenance, consent, licence, demographics, and split
  independence prevent conclusions about representativeness or leakage.
- The subjective `Good`/`Bad` labelling policy is not documented in a tracked
  measured report, so annotator preferences and cultural or situational bias
  may be encoded in the classifier.
- Performance is measured on one reported test set only. Results may change
  with lighting, pose, occlusion, image quality, camera source, or faces unlike
  those in that set; these conditions were not separately measured.
- Class-level errors remain material. The output must not be treated as a
  factual assessment of a person, identity, mood, attractiveness, or worth.
- Runtime speed has not been established by a reproducible tracked benchmark;
  it depends on hardware, image size, detection workload, and model state.

## Human Review and Recovery

Treat every classification as a review suggestion. A person should inspect the
original image and surrounding album context before removing anything,
especially when the prediction is uncertain or several people appear.

The application workflow should place removal candidates in its recoverable
`Trash` first. Review the Trash list before any permanent operating-system
deletion, and restore false positives from Trash. The model must not trigger
irreversible deletion without an explicit human decision.

## Model Artifact Integrity

The classifier asset named `mobilenet_face.pth` was downloaded from the current
GitHub release to a temporary directory and hashed on 2026-07-13:

- Release tag: `v1.0.0`
- Size: `12,098,423 bytes`
- SHA-256: `b7dd3b7d95c13c07167d269d65f49367d0b0007fcf0dc272ab1f94c34f3f4bf0`

The tracked evaluation report names `models/mobilenet_face.pth` but does not
record that file's digest. The available evidence therefore does not prove that
the published release checkpoint is byte-for-byte the checkpoint used to
produce the reported 75.1% result. Verify the release digest before use, and do
not treat an unverified checkpoint as the evaluated artifact.

`scripts/download_models.py` pins a separate MediaPipe face-detector TFLite
asset. Its digest is not a classifier-checkpoint digest and must not be replaced
with the value above.
