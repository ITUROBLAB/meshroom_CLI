# Meshroom implementation by David Castano for Windows. This adaptation of his code by Dr. Anirudh Rao (Imperial College London) to run on Linux machines. 
# Updated by Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v05

import sys
import os , os.path
from PIL import Image
#import shutil
import math
import time
from pathlib import Path
import json
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import numpy as np
from sklearn.preprocessing import StandardScaler

import open3d as o3d

from io import StringIO

dirname = os.path.dirname(os.path.abspath(__file__))  # Absolute path of this file

verboseLevel = 'error'  # detail of the logs (error, info, etc)

baseDir = ""


def SilentMkdir(theDir):    # function to create a directory
    try:
        os.mkdir(theDir)
    except:
        pass
    return 0

def run_0_downsampleImages(baseDir,imgDir,downsample_factor):

    taskFolder = '/0_DownsampledImages'
    outputDir = baseDir + taskFolder
    os.makedirs(outputDir, exist_ok=True)

    print("----------------------- 0/14 DOWNSAMPLING IMAGES -----------------------")

    if downsample_factor > 1.0:
        for filename in os.listdir(imgDir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(imgDir, filename)
                output_path = os.path.join(outputDir, filename)
                
                try:
                    with Image.open(input_path) as img:
                        # Calculate new dimensions
                        new_width = int(img.width / downsample_factor)
                        new_height = int(img.height / downsample_factor)
                        
                        # Downsample image
                        downsampled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Save downsampled image
                        downsampled_img.save(output_path)
                        print(f"Processed {filename} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

def run_1_cameraInit(binPath,baseDir,imgDir,downsample_factor):

    taskFolder = '/1_CameraInit'
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 1/14 CAMERA INITIALIZATION -----------------------")

    if downsample_factor > 1.0:
        imageFolder = baseDir + '/0_DownsampledImages' + "/"
    else:
        imageFolder = imgDir + "/"

    sensorDatabase = str(Path(binPath).parent) + '/share/aliceVision/cameraSensors.db' # Path to the sensors database, might change in later versions of meshroom
   
    output = baseDir + taskFolder + '/cameraInit.sfm'

    cmdLine = binPath + "aliceVision_cameraInit"
    cmdLine += " --imageFolder {0} --sensorDatabase {1} --output {2}".format(
        imageFolder, sensorDatabase, output)

    cmdLine += " --defaultFieldOfView 45" 
    cmdLine += " --allowSingleView 1"
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)

    return 0


def run_2_featureExtraction(binPath,baseDir , numberOfImages , imagesPerGroup=40, describerdensity="ultra",describerquality="ultra"):

    taskFolder = "/2_FeatureExtraction"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 2/14 FEATURE EXTRACTION -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    output = baseDir + taskFolder + "/"

    cmdLine = binPath + "aliceVision_featureExtraction"
    cmdLine += " --input {0} --output {1}".format(_input, output)
    cmdLine += " --forceCpuExtraction 0"


    #when there are more than 40 images, it is good to send them in groups
    if(numberOfImages>imagesPerGroup):
        numberOfGroups=int(math.ceil( numberOfImages/imagesPerGroup))
        for i in range(numberOfGroups):
            cmd=cmdLine + " --rangeStart {} --rangeSize {} ".format(i*imagesPerGroup,imagesPerGroup)
            print("------- group {} / {} --------".format(i+1,numberOfGroups))
            print(cmd)
            os.system(cmd)

    else:
        print(cmdLine)
        os.system(cmdLine)


def run_3_imageMatching(binPath,baseDir):

    taskFolder = "/3_ImageMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 3/14 IMAGE MATCHING -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    featuresFolders = baseDir + '/2_FeatureExtraction' + "/"
    output = baseDir + taskFolder + '/imageMatches.txt'

    cmdLine = binPath + "aliceVision_imageMatching"
    cmdLine += " --input {0} --featuresFolders {1} --output {2}".format(
        _input, featuresFolders, output)

    cmdLine +=  " --tree " + "/"+ str(Path(binPath).parent)+ "/share/aliceVision/vlfeat_K80L3.SIFT.tree/"
    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_4_featureMatching(binPath,baseDir,numberOfImages,imagesPerGroup=20):

    taskFolder = "/4_featureMatching"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 4/14 FEATURE MATCHING -----------------------")

    _input =   baseDir + '/1_CameraInit/cameraInit.sfm'
    output =  baseDir + taskFolder + "/"
    featuresFolders =  baseDir + '/2_FeatureExtraction' + "/"
    imagePairsList =  baseDir + '/3_ImageMatching/imageMatches.txt'

    cmdLine = binPath + "aliceVision_featureMatching"
    cmdLine += " --input {0} --featuresFolders {1} --output {2} --imagePairsList {3}".format(
        _input, featuresFolders, output, imagePairsList)

    cmdLine += " --knownPosesGeometricErrorMax 5"
    cmdLine += " --verboseLevel " + verboseLevel

    cmdLine += " --describerTypes sift --photometricMatchingMethod ANN_L2 --geometricEstimator acransac --geometricFilterType fundamental_matrix --distanceRatio 0.8"
    cmdLine += " --maxIteration 2048 --geometricError 0.0 --maxMatches 0"
    cmdLine += " --savePutativeMatches False --guidedMatching False --matchFromKnownCameraPoses False --exportDebugFiles True"

    #when there are more than 20 images, it is good to send them in groups
    if(numberOfImages>imagesPerGroup):
        numberOfGroups=math.ceil( numberOfImages/imagesPerGroup)
        for i in range(numberOfGroups):
            cmd=cmdLine + " --rangeStart {} --rangeSize {} ".format(i*imagesPerGroup,imagesPerGroup)
            print("------- group {} / {} --------".format(i,numberOfGroups))
            print(cmd)
            os.system(cmd)

    else:
        print(cmdLine)
        os.system(cmdLine)

def run_5_structureFromMotion(binPath,baseDir):

    taskFolder = "/5_structureFromMotion"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 5/14 STRUCTURE FROM MOTION -----------------------")

    _input = baseDir + '/1_CameraInit/cameraInit.sfm'
    output = baseDir + taskFolder + '/sfm.abc' 
    outputViewsAndPoses = baseDir + taskFolder + '/cameras.sfm' 
    extraInfoFolder = baseDir + taskFolder + "/"
    featuresFolders = baseDir + '/2_FeatureExtraction' + "/"
    matchesFolders =  baseDir + '/4_featureMatching' + "/"

    cmdLine = binPath + "aliceVision_incrementalSfM"
    cmdLine += " --input {0} --output {1} --outputViewsAndPoses {2} --extraInfoFolder {3} --featuresFolders {4} --matchesFolders {5}".format(
        _input, output, outputViewsAndPoses, extraInfoFolder, featuresFolders, matchesFolders)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_6_prepareDenseScene(binPath,baseDir):
    taskFolder = "/6_PrepareDenseScene"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 6/14 PREPARE DENSE SCENE -----------------------")
    _input = baseDir +  '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/" 

    cmdLine = binPath + "aliceVision_prepareDenseScene"
    cmdLine += " --input {0}  --output {1} ".format(_input,  output)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_7_depthMap(binPath,baseDir ,numberOfImages , groupSize=6 , downscale = 2):
    taskFolder = "/7_DepthMap"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 7/14 DEPTH MAP -----------------------")
    _input = baseDir +   '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/"
    imagesFolder = baseDir + '/6_PrepareDenseScene' + "/"

    cmdLine = binPath + "aliceVision_depthMapEstimation"
    cmdLine += " --input {0}  --output {1} --imagesFolder {2}".format(
        _input,  output, imagesFolder)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --downscale " + str(downscale)
    
    numberOfBatches = int(math.ceil( numberOfImages / groupSize ))

    for i in range(numberOfBatches):
        groupStart = groupSize * i
        currentGroupSize = min(groupSize,numberOfImages - groupStart)
        if groupSize > 1:
            print("DepthMap Group {} of {} : {} to {}".format(i, numberOfBatches, groupStart, currentGroupSize))
            cmd = cmdLine + (" --rangeStart {} --rangeSize {}".format(str(groupStart),str(groupSize)))       
            print(cmd)
            os.system(cmd)


def run_8_depthMapFilter(binPath,baseDir):
    taskFolder = "/8_DepthMapFilter"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 8/14 DEPTH MAP FILTER-----------------------")
    _input = baseDir +   '/5_structureFromMotion/sfm.abc'
    output = baseDir + taskFolder + "/"
    depthMapsFolder = baseDir + '/7_DepthMap' + "/"

    cmdLine = binPath + "aliceVision_depthMapFiltering"
    cmdLine += " --input {0}  --output {1} --depthMapsFolder {2}".format(
        _input,  output, depthMapsFolder)

    cmdLine += " --verboseLevel " + verboseLevel

    print(cmdLine)
    os.system(cmdLine)


def run_9_meshing(binPath,baseDir  , maxInputPoints = 500000000  , maxPoints=100000000, colorizeoutput="True"):
    taskFolder = "/9_Meshing"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 9/14 MESHING -----------------------")
    _input = baseDir +  '/5_structureFromMotion/sfm.abc'
    output = baseDir +   taskFolder + '/densePointCloud.abc'
    outputMesh = baseDir + taskFolder + '/mesh.obj' 
    depthMapsFolder = baseDir + '/8_DepthMapFilter' + "/"

    cmdLine = binPath + "aliceVision_meshing"
    cmdLine += " --input {0}  --output {1} --outputMesh {2} --depthMapsFolder {3} ".format(
        _input,  output, outputMesh, depthMapsFolder)

    cmdLine += " --maxInputPoints " + str(maxInputPoints)
    cmdLine += " --maxPoints " + str(maxPoints)
    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --colorizeOutput " + colorizeoutput

    print(cmdLine)
    os.system(cmdLine)

def run_14_convertSFMFormat(binPath,baseDir  , SFMFileFormat = "ply"):
    taskFolder = "/14_convertSFMFormat"
    describers = "unknown"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 14/14 CONVERTING -----------------------")
    _input = baseDir +   '/9_Meshing/densePointCloud.abc'
    output = baseDir +   taskFolder + '/densePointCloud.ply'

    cmdLine = binPath + "aliceVision_convertSfMFormat"
    cmdLine += " --input {0}  --output {1} ".format(
        _input,  output)

    cmdLine += " --verboseLevel " + verboseLevel

    cmdLine += " --describerTypes " + describers


    print(cmdLine)
    os.system(cmdLine)


def run_10_meshFiltering(binPath,baseDir ,keepLargestMeshOnly="True", smoothingiterations=100):
    taskFolder = "/10_MeshFiltering"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 10/14 MESH FILTERING -----------------------")
    inputMesh = baseDir + '/9_Meshing/mesh.obj' 
    outputMesh = baseDir + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshFiltering"
    cmdLine += " --inputMesh {0}  --outputMesh {1}".format(
        inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --keepLargestMeshOnly " + keepLargestMeshOnly

    print(cmdLine)
    os.system(cmdLine)


def run_11_meshDecimate(binPath,baseDir , simplificationFactor=0.0 , maxVertices=100000):
    taskFolder = "/11_MeshDecimate"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 11/14 MESH DECIMATE -----------------------")
    inputMesh = baseDir + '/10_MeshFiltering/mesh.obj'
    outputMesh = baseDir + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshDecimate"
    cmdLine += " --input {0}  --output {1}".format(
        inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --simplificationFactor " + str(simplificationFactor)
    cmdLine += " --maxVertices " + str(maxVertices)

    print(cmdLine)
    os.system(cmdLine)


def run_12_meshResampling(binPath,baseDir , simplificationFactor=0.0 , maxVertices=100000):
    taskFolder = "/12_MeshResampling"
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 12/14 MESH RESAMPLING -----------------------")
    inputMesh = baseDir +  '/11_MeshDecimate/mesh.obj' 
    outputMesh = baseDir  + taskFolder + '/mesh.obj'

    cmdLine = binPath + "aliceVision_meshResampling"
    cmdLine += " --input {0}  --output {1}".format( inputMesh, outputMesh)

    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --simplificationFactor " + str(simplificationFactor)
    cmdLine += " --maxVertices " + str(maxVertices)

    print(cmdLine)
    os.system(cmdLine)


def run_13_texturing(binPath , baseDir , textureSide = 16384 , downscale=1 , unwrapMethod = "Basic", fillholes="True"):
    taskFolder = '/13_Texturing'
    SilentMkdir(baseDir + taskFolder)

    print("----------------------- 13/14 TEXTURING  -----------------------")
    _input = baseDir +   '/9_Meshing/densePointCloud.abc' 
    imagesFolder = baseDir + '/6_PrepareDenseScene' "/"
    inputMesh = baseDir + '/12_MeshResampling/mesh.obj'
    output = baseDir + taskFolder + "/"

    cmdLine = binPath + "aliceVision_texturing"
    cmdLine += " --input {0} --inputMesh {1} --output {2} --imagesFolder {3}".format(
        _input, inputMesh, output, imagesFolder)

    cmdLine += " --textureSide " + str(textureSide)
    cmdLine += " --downscale " + str(downscale)
    cmdLine += " --verboseLevel " + verboseLevel
    cmdLine += " --unwrapMethod " + unwrapMethod

    print(cmdLine)
    os.system(cmdLine)


# Add ImageMatchingMultiSfM for recursive update 


def main():

    first_iteration_ = True
    global baseDir

    # Pass the arguments of the function as parameters in the command line code
    binPath = sys.argv[1]           ##  --> path of the binary files from Meshroom
    baseDir = sys.argv[2]           ##  --> name of the Folder containing the process (a new folder will be created)
    imgDir = sys.argv[3]            ##  --> Folder containing the images 
    if len(sys.argv) > 4:
        downsample_factor = float(sys.argv[4]) ##  --> To downsample input images 
    else:
        downsample_factor = 1.0
    rerunCylce = 5 ## --> Reconstruction rerun every ... number of new images

    SilentMkdir(baseDir)

    try:
        while True:

            startTime = time.time()

            numberOfImages = len([name for name in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, name))])

            if first_iteration_:
                if numberOfImages > rerunCylce:
                    print("Images found in the directory. Starting Meshroom processing.")
                    updateProcess(binPath, imgDir, numberOfImages, downsample_factor)
                    scalePC()
                    endTime = time.time()
                    hours, rem = divmod(endTime-startTime, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print("time elapsed: "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                first_iteration_ = False
            elif checkForNewImages(numberOfImages, rerunCylce):
                print("New images detected. Updating the Meshroom processing steps.")
                updateProcess(binPath, imgDir, numberOfImages, downsample_factor)
                scalePC()
                endTime = time.time()
                hours, rem = divmod(endTime-startTime, 3600)
                minutes, seconds = divmod(rem, 60)
                print("time elapsed: "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            else:
                print("No new images detected. Waiting for changes...")
                time.sleep(5)  # 5 seconds wait for new check
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")

def checkForNewImages(currentCount, rerunLimit):
    global baseDir
    lastCountFile = os.path.join(baseDir, "last_image_count.txt")

    if os.path.exists(lastCountFile):
        with open(lastCountFile, "r") as f:
            lastCount = int(f.read().strip())
        if currentCount >= lastCount + rerunLimit:
            with open(lastCountFile, "w") as f:
                f.write(str(currentCount))
            return True
        else:
            return False
    else:
        # First run, create the last count file
        with open(lastCountFile, "w") as f:
            f.write(str(currentCount))
        return False

def parse_estimated_poses(sfm_file):
    with open(sfm_file, 'r') as f:
        data = json.load(f)
    poses = {}

    # Create a mapping from poseId to image name
    poseid_to_imagename = {}
    for view in data['views']:
        pose_id = view['poseId']
        image_path = view['path']
        image_name = os.path.basename(image_path)  # Extract the image name from the full path
        poseid_to_imagename[pose_id] = image_name

    # Parse the poses and add the image name to each
    for pose in data['poses']:
        pose_id = pose['poseId']
        rotation = pose['pose']['transform']['rotation']
        center = pose['pose']['transform']['center']
        image_name = poseid_to_imagename.get(pose_id, "Unknown")  # Get the image name for the poseId
        poses[pose_id] = {
            'rotation': [float(r) for r in rotation],
            'center': [float(c) for c in center],
            'image_name': image_name  # Add the image name
        }

    return poses

def parse_real_poses(csv_file):
    # Assuming the order: image_name, x, y, z, roll, pitch, yaw
    df = pd.read_csv(csv_file, header=None, names=['image_name', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'], sep=',')
    poses = {}
    for index, row in df.iterrows():
        image_name = row['image_name']
        position = row[['x', 'y', 'z']].tolist()
        orientation = row[['roll', 'pitch', 'yaw']].tolist()
        poses[image_name] = {
            'position': [float(p) for p in position],
            'orientation': [float(o) for o in orientation]
        }
    return poses

def rpy_to_rotation_matrix(rpy):
    r = R.from_euler('xyz', rpy)
    return r.as_matrix().flatten().tolist()

def find_affine_transformation(points, target_points):
    """
    Find the affine transformation that maps points to target_points.
    :param points: numpy array of shape (N, 3) representing the points
    :param target_points: numpy array of shape (N, 3) representing the target points
    :return: The affine parameters as numpy array of shape (12,)
    """
    initial_params = np.zeros(12)
    result = least_squares(residuals, initial_params, args=(points, target_points))
    return result.x

def affine_transformation(points, affine_params):
    """
    Apply an affine transformation to the given points.
    :param points: numpy array of shape (N, 3) representing the points
    :param affine_params: numpy array of shape (12,) representing the affine parameters (9 for the matrix and 3 for the translation)
    :return: Transformed points as numpy array of shape (N, 3)
    """
    A = affine_params[:9].reshape(3, 3)
    b = affine_params[9:]
    return np.dot(points, A.T) + b

def residuals(affine_params, points, target_points):
    """
    Compute residuals for the affine transformation fitting.
    :param affine_params: numpy array of shape (12,) representing the affine parameters
    :param points: numpy array of shape (N, 3) representing the points
    :param target_points: numpy array of shape (N, 3) representing the target points
    :return: Residuals as numpy array of shape (N*3,)
    """
    transformed_points = affine_transformation(points, affine_params)
    return (transformed_points - target_points).flatten()

def updateProcess(binPath, imgDir, numberOfImages, downsample_factor=1.0):
    #print("No need to update")

    run_0_downsampleImages(baseDir,imgDir,downsample_factor)
    run_1_cameraInit(binPath,baseDir,imgDir,downsample_factor)
    run_2_featureExtraction(binPath,baseDir, numberOfImages)
    run_3_imageMatching(binPath,baseDir)
    run_4_featureMatching(binPath,baseDir,numberOfImages)
    run_5_structureFromMotion(binPath,baseDir)
    run_6_prepareDenseScene(binPath,baseDir)
    run_7_depthMap(binPath,baseDir , numberOfImages )
    run_8_depthMapFilter(binPath,baseDir)
    run_9_meshing(binPath,baseDir)
    #run_10_meshFiltering(binPath,baseDir)
    #run_11_meshDecimate(binPath,baseDir)
    #run_12_meshResampling(binPath,baseDir)
    #run_13_texturing(binPath,baseDir)
    run_14_convertSFMFormat(binPath,baseDir)

def scalePC():
    taskFolder = "/5_structureFromMotion"
    outputViewsAndPoses = baseDir + taskFolder + '/cameras.sfm' 
    estimated_poses = parse_estimated_poses(outputViewsAndPoses)

    #real_poses = parse_real_poses('/home/ayilmaz/AGRI-Opencore/DataSet_Franka_Arm/20240205_trial04_all_lattices_in_an_order/image_info_meshroom_trial.csv')
    real_poses = parse_real_poses('/home/ayilmaz/AGRI-Opencore/DataSet_Franka_Arm/20240711_all/image_info_Meshroom.csv')

    print("Real poses: ", real_poses)

    for image_name, pose in real_poses.items():
        orientation = pose['orientation']
        #print(f"Converting RPY for {image_name}: {orientation}")
        real_poses[image_name]['rotation_matrix'] = rpy_to_rotation_matrix(pose['orientation'])

    # Collect corresponding points
    estimated_points = []
    real_points = []

    for image_name, real_pose in real_poses.items():
        for pose_id, estimated_pose in estimated_poses.items():
            #print("estimated_pose['image_name']: ", estimated_pose['image_name'])
            if image_name == estimated_pose['image_name']:
                estimated_points.append(estimated_pose['center'])
                real_points.append(real_pose['position'])

    P = np.array(estimated_points)
    Q = np.array(real_points)

    # Find the optimal transformation
    affine_params = find_affine_transformation(P, Q)

    print("Affine transformation: ", affine_params)

    # Apply the affine transformation to the estimated points
    transformed_points = affine_transformation(P, affine_params)

    # Print out transformed points and real points to see the similarity
    for i in range(len(Q)):
        print(f"Real Point {i}: {Q[i]}")
        print(f"Transformed Point {i}: {transformed_points[i]}")
        print(f"Estimated Point {i}: {P[i]}")

    taskFolder = "/14_convertSFMFormat"
    _input = baseDir +   taskFolder + '/densePointCloud.ply'
    output = baseDir + taskFolder + '/densePointCloud_tf.ply'

    point_cloud = o3d.io.read_point_cloud(_input)

    # Apply the affine transformation to the point cloud
    transformed_point_cloud = apply_affine_transformation_to_point_cloud(point_cloud, affine_params)

    # Save the transformed point cloud to the output file
    o3d.io.write_point_cloud(output, transformed_point_cloud)

    print("Transformed PC saved")

def apply_affine_transformation_to_point_cloud(point_cloud, affine_params):
    points = np.asarray(point_cloud.points)
    transformed_points = affine_transformation(points, affine_params)
    
    # Create a new point cloud for the transformed points
    transformed_point_cloud = o3d.geometry.PointCloud()
    transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

    # Preserve RGB colors if available
    if point_cloud.has_colors():
        transformed_point_cloud.colors = point_cloud.colors

    # Preserve other attributes if available
    if point_cloud.has_normals():
        transformed_point_cloud.normals = point_cloud.normals

    return transformed_point_cloud

if __name__ == "__main__":
    main()
