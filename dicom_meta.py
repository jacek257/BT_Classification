import argparse
import os
import pydicom
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool

FIELDS = [
    'AccessionNumber',
    'AcquisitionMatrix',
    'B1rms',
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'ConversionType',
    'DiffusionBValue',
    'DiffusionGradientOrientation',
    'EchoNumbers',
    'EchoTime',
    'EchoTrainLength',
    'FlipAngle',
    'HighBit',
    'HighRRValue',
    'ImageDimensions',
    'ImageFormat',
    'ImageGeometryType',
    'ImageLocation',
    'ImageOrientation',
    'ImageOrientationPatient',
    'ImagePosition',
    'ImagePositionPatient',
    'ImageType',
    'ImagedNucleus',
    'ImagingFrequency',
    'InPlanePhaseEncodingDirection',
    'InStackPositionNumber',
    'InstanceNumber',
    'InversionTime',
    'Laterality',
    'LowRRValue',
    'MRAcquisitionType',
    'MagneticFieldStrength',
    'Modality',
    'NumberOfAverages',
    'NumberOfPhaseEncodingSteps',
    'PatientID',
    'PatientName',
    'PatientPosition',
    'PercentPhaseFieldOfView',
    'PercentSampling',
    'PhotometricInterpretation',
    'PixelBandwidth',
    'PixelPaddingValue',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'PositionReferenceIndicator',
    'PresentationLUTShape',
    'ReconstructionDiameter',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SAR',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SeriesDescription',
    'SeriesInstanceUID',
    'SeriesNumber',
    'SliceLocation',
    'SliceThickness',
    'SpacingBetweenSlices',
    'SpatialResolution',
    'SpecificCharacterSet',
    'StudyInstanceUID',
    'TemporalResolution',
    'TransferSyntaxUID',
    'TriggerWindow',
    'WindowCenter',
    'WindowWidth'
]

FM_FIELDS = [
    'FileMetaInformationGroupLength',
    'FileMetaInformationVersion',
    'ImplementationClassUID',
    'ImplementationVersionName',
    'MediaStorageSOPClassUID',
    'MediaStorageSOPInstanceUID',
    'SourceApplicationEntityTitle',
    'TransferSyntaxUID',
]

final = []


def get_meta_info(dicom):
    row = {f: dicom.get(f) for f in FIELDS}
    row_fm = {f: dicom.file_meta.get(f) for f in FM_FIELDS}
    row_other = {
        'is_original_encoding': dicom.is_original_encoding,
        'is_implicit_VR': dicom.is_implicit_VR,
        'is_little_endian': dicom.is_little_endian,
        'timestamp': dicom.timestamp,
    }
    return {**row, **row_fm, **row_other}


def get_dicom_files(input_dir, ds='train'):
    dicoms = []

    for subdir, dirs, files in os.walk(f"{input_dir}/{ds}"):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".dcm"):
                dicoms.append(filepath)

    return dicoms


def process_dicom(dicom_src, _x):
    dicom = pydicom.dcmread(dicom_src)
    file_data = dicom_src.split("/")
    file_src = "/".join(file_data[-4:])

    tmp = {"BraTS21ID": file_data[-3], "dataset": file_data[-4], "type": file_data[-2], "dicom_src": f"./{file_src}"}
    tmp.update(get_meta_info(dicom))

    return tmp


def update(res):
    if res is not None:
        final.append(res)

    pbar.update()


def error(e):
    print(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="./input")
    ap.add_argument("--output", type=str, default="./")
    ap.add_argument("--dataset", type=str, default="train")
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--debug", type=int, default=0)

    args = vars(ap.parse_args())

    dicom_files = get_dicom_files(args["input"], args["dataset"])

    if args["debug"]:
        dicom_files = dicom_files[:1000]

    pool = Pool(processes=args["n_jobs"])
    pbar = tqdm(total=len(dicom_files))

    for dicom_file in dicom_files:
        pool.apply_async(
            process_dicom,
            args=(dicom_file, ''),
            callback=update,
            error_callback=error,
        )

    pool.close()
    pool.join()
    pbar.close()

    final = pd.DataFrame(final)
    final.to_csv(f"{args['output']}/dicom_meta_{args['dataset']}.csv", index=False) 
    final.to_pickle(f"{args['output']}/dicom_meta_{args['dataset']}.pickle")
