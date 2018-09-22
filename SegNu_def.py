# David Garcia Seisdedos
# SegNu
# Definitions
# May 2018

#####################################
# Load libraries
#####################################
import sys
import os
import subprocess as sp
import numpy as np
import itertools


from skimage import io, color
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage.morphology import closing, square

from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as k

from scipy import ndimage as ndi

from PIL import Image
import time
####################################
# Definitions
####################################

def apply_watershed (binary_image, label_image, min_distance, raw_image, cluster_regions):
    """
    Apply watershed segmentation.

    :param binary_image: A numpy matrix. Binary image.
    :param label_image: A numpy matrix. Matrix label image.
    :param min_distance: A float value. Minimum numberof pixels separating local maximum peaks.
    :param raw_image: A numpy matrix. Original image.
    :param cluster_regions: A list with cluster regions.
    """
    def is_region_in_peaks(cluster_regions, peaks,boolean_peaks):
        if cluster_regions == []:
            return (boolean_peaks)
        else:
            tmp_region = cluster_regions[0].bbox
            for i in peaks:
                if tmp_region[0]< i[0] <tmp_region[2] and tmp_region[1]< i[1] <tmp_region[3]:
                    return (is_region_in_peaks(cluster_regions[1:], peaks,boolean_peaks))
       
            boolean_peaks[int(cluster_regions[0].centroid[0]),int(cluster_regions[0].centroid[1])]=True
            return (is_region_in_peaks(cluster_regions[1:], peaks, boolean_peaks))
        
    distance = ndi.distance_transform_edt(binary_image)
    boolean_peaks = peak_local_max(distance,
                                indices=False,
                                min_distance=min_distance,
                                labels=label_image)
    peaks = peak_local_max(distance,
                                indices=True,
                                min_distance=min_distance,
                                labels=label_image)
    local_maxi = is_region_in_peaks(cluster_regions, peaks,boolean_peaks)
    markers = ndi.label(local_maxi)[0]
    label_image = watershed(-distance, markers=markers, mask=binary_image)
    label_image *= binary_image
    regions = regionprops(label_image, raw_image)
    return (regions, label_image)

def generate_patches(regions, box, label_image, original_image, CELL_NUM=0, original_patch_size=192):
    """
    Generate patches with the input regions.

    :param regions: A list of scikit-image regions.
    :param box: An integer value. Number of the pixels in the side of the patch.
    :param label_image: A numpy matrix. Labeled image.
    :param original_image: A numpy matrix. Original image.
    :param CELL_NUM: A integer value. Cell counter.
    :param original_patch_size: A numpy matrix. Original image.
    """
    input_shape = original_image.shape
    region_image = label_image.copy()
    for region in regions:
        region_image[:,:] = 0
        region_image[label_image == region.label] = region.label
        region_image[region_image != 0] = original_image[region_image != 0]
        centre = region.centroid
        bbox = [(int(centre[0]-box/2)), (int(centre[0]+box/2)), (int(centre[1]-box/2)), (int(centre[1]+box/2))]
        for i,j in enumerate(bbox):
            if j < 0:
                bbox[i+1] = box
                bbox[i] = 0
            if j > input_shape[0] and i == 1:
                bbox[i-1] = input_shape[0] - box
                bbox[i] = input_shape[0]
            if j > input_shape[1] and i == 3:
                bbox[i-1] = input_shape[1] - box
                bbox[i] = input_shape[1]
                          
        CELL_NUM += 1
        sel_cel = region_image[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        sel_cel = resize(sel_cel, (original_patch_size,original_patch_size), mode="symmetric", preserve_range=True)
        sel_cel = sel_cel*255/sel_cel.max()
        col_sel_cel = np.matrix(sel_cel, "uint8")
        img = Image.fromarray(col_sel_cel)
        img.save("".join(["/tmp/SegNu/Samples/",str(CELL_NUM+1000000),".tif"]))
        
    return 

def CNN_inference(dim_patch, num_regions, argmax = True):
    """
    Generate patches with the input regions.

    :param dim_patch: An integer value. Number of the pixels in the side of the patch.
    :param num_regions: An integer value. Number of the region to be analyse.
    :param argmax: Wether get the inferred class for each region (True) or get the softmax prediction value for a region with a single cell.
    """
    nb_classes = 3
    img_width, img_height = dim_patch,dim_patch
    batch_size = 16
    n = 16
    samples_data_dir = '/tmp/SegNu'
    model_pretrained = './weights/model_weights.h5'
    base_model = VGG16(input_shape = (dim_patch, dim_patch, 3), weights = 'imagenet', include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    model.load_weights(model_pretrained)
    predict_data = ImageDataGenerator(rescale=1. / 255)
    validation_generator = predict_data.flow_from_directory(samples_data_dir,
                                                            target_size = (img_width, img_height),
                                                            batch_size = batch_size,
                                                            class_mode = None,
                                                            shuffle = False)
    predictions = model.predict_generator(validation_generator)
    k.clear_session()
    if argmax==False:
        return (predictions[:,1])
    else:
        try:
            predictions = predictions.argmax(axis=1).tolist()
            return (predictions)
        except AttributeError:
            return ([0])

def set_arg(argv):
    """
    Set the input and output arguments

    :param argv: Input arguments.
    """
    import getopt
    outputpath = ""
    segmentation = "watershed"
    save_images = False
    try:
        opts, args = getopt.getopt(argv,"hi:o:t:s",["ipath=","opath=",  "save_images"])
    except getopt.GetoptError:
        print ("ERROR")
        print ('SegNu.py -i <inputpath> -o <outputpath> -s <save_images>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('\nSegNu.py -i <inputpath> -o <outputpath> -s <save_images> -t <type_of_segmentation>\n\n<inputpath> path to the working images directory\n<outputpath> path to the output file\n<save_images>wether or not save the processed images')
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputpath = arg
        elif opt in ("-o", "--opath"):
            outputpath = arg
        elif opt in ("-s", "--save_images"):
            save_images = True
    if outputpath == "":
        outputpath = inputpath + "/Results"
    print ("Input path is " + inputpath)
    print ("Output path is " + outputpath)
    print ("Save images? " + str(save_images))
    return (inputpath, outputpath, save_images)

def create_folders(output_folder):
     """
     Create the working folders.
     """
     sp.call([" ".join(["mkdir","/tmp/SegNu"])],shell=True)
     sp.call([" ".join(["mkdir","/tmp/SegNu/Samples"])],shell=True)
     sp.call([" ".join(["mkdir",output_folder])],shell=True)
     sp.call([" ".join(["mkdir",output_folder+"/Instances_images"])],shell=True)
     sp.call([" ".join(["mkdir",output_folder+"/Modified_images"])],shell=True)


def rm_tmp_files():
     """
     Remove temporary files.
     """
     sp.call([" ".join(["rm","/tmp/SegNu/Samples/*"])], shell=True)

def run_watershed_seg(radio,
                      segmented_regions,
                      cluster_regions,
                      label_image,
                      image_grey,
                      patch,
                      threshold,
                      num_split=1):
    """
    Run watershed segmentation recursively.

    :param radio: An integer value. Expected cell radio. 
    :param segmented_regions: A list of unicell scikit-image regions.
    :param cluster_regions: A list of cells clusters scikit-image regions.
    :param region_image: A numpy matrix. Labeled image.
    :param image_grey: A numpy matrix. Original image.
    :param patch: An integer value. Leght (in pixels) of the side of the sqare patch.
    """
    from random import shuffle
    from skimage.future import graph
    if cluster_regions == [] or num_split>=10:
        segmented_regions += cluster_regions
        segmented_image = np.zeros((label_image.shape[0], label_image.shape[1]))
        
        for h,i in enumerate(segmented_regions):
            i.label = h+1
            for j in range(i.coords.shape[0]):
                segmented_image[i.coords[j,0],i.coords[j,1]] = i.label
        rag = graph.RAG(segmented_image)
        try:
            rag.remove_node(0)
        except networkx.exception.NetworkXError:
            print( "Not node 0")
        for n,d in rag.nodes_iter(data=True):
            d['labels']=[n]
        for x,y,d in rag.edges_iter(data=True):
            d['weight']=1
        regions_to_evaluate = [n for n in rag if len(rag[n])>=1]
        
        final_proposes=get_optimized_cells(rag, regions_to_evaluate, segmented_regions,patch,segmented_image, image_grey )
        
        
        
        lb = label_final_cell_proposed(final_proposes, segmented_image)
        del(rag, final_proposes, regions_to_evaluate)
        final_regions,instances_image=sort_instance_image(lb, image_grey)
        
        instances_image[:,:] = 0
        
        shuffle(final_regions)
        for h,i in enumerate(final_regions):
            i.label=h+1
            for j in range(i.coords.shape[0]):
                instances_image[i.coords[j,0],i.coords[j,1]]=i.label
        return ( instances_image, final_regions)
    else:
        region_image = np.zeros((label_image.shape[0],label_image.shape[1]))
        image_leftovers =region_image.copy()
        image_labels=region_image.copy()
        for h,j in enumerate(cluster_regions):
            region_image[label_image == j.label] = j.label
            image_labels[label_image == j.label] = j.label
            image_leftovers[region_image != 0] = image_grey[region_image != 0]
            region_image[:,:] = 0
           
        bw = closing(image_leftovers > threshold, square(3))
        
        regions_splitted, label_image = apply_watershed(bw, image_labels,radio/num_split,image_leftovers, cluster_regions)
        generate_patches(regions_splitted,patch, label_image,image_grey,CELL_NUM=1000*num_split)
        
        
        predictions = CNN_inference(192,len(regions_splitted))
        segmented_regions+=[j for i, j in enumerate(regions_splitted) if predictions[i]<=1 ]
        cluster_regions= [j for i,j in enumerate(regions_splitted) if predictions[i]==2]
        
        rm_tmp_files()
        num_split+=1
        return (run_watershed_seg(radio, segmented_regions, cluster_regions, label_image, image_grey, patch, threshold, num_split))


def sort_instance_image(region_image,im_grey):
    tmp_im = region_image.copy()
    tmp_im_final = region_image.copy()
    tmp_im_final[:,:]=0
    final_regions = []
    for i,j in enumerate(np.unique(region_image)[1:]):
        
        tmp_im[:,:] = 0
        tmp_im[region_image == j] = i+1
        tmp_im_final[region_image == j] = i+1
        bw = closing(tmp_im > 0, square(3))
        label_image = label(bw)
        reg_pp=regionprops(label_image, im_grey)[0]
        a=reg_pp.coords
        reg_pp.label=i+1
        a=reg_pp.coords
        
        final_regions+=[reg_pp]
    return (final_regions,tmp_im_final ) 

def initial_segmentation(image,
                         PATCH_RATIO):
    """
    Initialization of the image segmentation.
 
    :param image: A numpy matrix. Original image.
    :param PATCH_RATIO: Constant. Ratio between patch area against cell mean cell area.
    """
    from skimage.filters import threshold_li, threshold_otsu, threshold_minimum
    from skimage.morphology import closing, square
    thresh_li = threshold_li(image)
    thresh_otsu = threshold_otsu(image)
    try:
        thresh_min = threshold_minimum(image)
    except RuntimeError:
        thresh_min = thresh_otsu + 100

    if thresh_min < thresh_otsu:
        threshold_steps = [thresh_li]
    else:
        threshold_steps = [thresh_otsu]
    binary_image = closing(image > threshold_steps[0], square(3))
    label_image = label(binary_image)
    regions = regionprops(label_image, image)
    regions_above_noise = []
    areas = []
    for region in regions:
        if region.area >= 9:
            #Lista con las nuevas regiones
            areas.append(region.area)
            regions_above_noise.append(region)

    median=np.median(areas)
    patch = int((PATCH_RATIO*median)**(0.5))
    return (patch, regions_above_noise, label_image, threshold_steps[0])

def save_mod_images(raw_image,segmented_image, unicell_regions,output_dir, file):
    """
    Saving the segmented image over the orignal image
 
    :param raw_image: A numpy matrix. Original image.
    :param segmented_image: A numpy matrix. Labeled image.
	:param unicell_regions: Constant. A list of unicell scikit-image regions.
	:param output_dir: String. Output directory.
	:param file: String. Input image path.
    """
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    fig, ax = plt.subplots(figsize=(10,10))
    CELL_NUM = 1
    label_image=segmented_image.copy()
    for region in unicell_regions:
        segmented_image[:,:] = 0
        segmented_image[label_image == region.label] = region.label
        contour=find_contours(segmented_image,0.5)
        count = [i.shape for i in contour]
        try:
            contour=contour[count.index(max(count))]
            ax.plot(contour[:,1], contour[:,0],'r', linewidth=1)
            minr, minc, maxr, maxc = region.bbox
            ax.text(0.5*(minc+maxc),0.5*(minr+maxr), str(CELL_NUM), horizontalalignment="center", verticalalignment="center", fontsize=10, color="blue")
            CELL_NUM += 1
        except ValueError:
            print ("Not contour found")
    ax.imshow(raw_image, cmap='gray')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("".join([output_dir,"/Modified_images/",file.split(".")[0],".tif"]), dpi=fig.dpi,bbox_inches="tight")
    return



def get_minimum_linked_region(rag, subgraph):
    tmp=0
    link = 1
    while tmp!=0:
        single_union = [n for n in rag if len(rag[n])==link and n in subgraph][0]
        if single_union != []:
            tmp=single_union
    return(tmp)

def get_unique_subgraph(unique_regions, rag, subgraph=[]):
    
    if unique_regions == []:
        return (list(subgraph))
    else:
        
        tmp =list(rag[unique_regions[0]])
        unique_regions=unique_regions[1:]
        for i in tmp:
            if i not in subgraph:
                subgraph+=[i]
                unique_regions+=[i]
        return (get_unique_subgraph(unique_regions,  rag, subgraph))  

def get_subgraphs (rag, regions_to_evaluate, subgraphs=[]):
    if regions_to_evaluate == []:
        return (subgraphs)
    else:
        tmp = regions_to_evaluate[0]
        tmp_graph = get_unique_subgraph([tmp], rag,subgraph=[])
        regions_to_evaluate = [i for i in regions_to_evaluate if i not in tmp_graph]
        if len(tmp_graph)>1:
            subgraphs.append(tmp_graph)
        return (get_subgraphs(rag, regions_to_evaluate, subgraphs))

def get_most_tied_region(rag, subgraph, regions_included):
    num_ties = []
    sg =subgraph[:]
    for i in sg:
        if i in regions_included:
            num_ties.append(len(rag[i]))
        else:
            num_ties.append(0)
    most_tied_region=sg.pop(num_ties.index(max(num_ties)))
    single_union = rag[most_tied_region]
    return(most_tied_region,list(single_union) )

def combinations(iterable, r, constant):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    tmp = [pool[i] for i in indices] + [constant]
    
    yield list(tmp)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        tmp = [pool[i] for i in indices] + [constant]
        
        yield list(tmp)

def get_regions_combination(rag, regions_to_evaluate_dict, regions_included):
    sg = regions_to_evaluate_dict.keys()
    
    most_tied_region, sg_without_mtr = get_most_tied_region(rag, sg, regions_included)
    
    sg_combi = [regions_to_evaluate_dict[most_tied_region]]
    
    sg_combi_redux = [regions_to_evaluate_dict[most_tied_region]]
    for i in range(len(sg_without_mtr)):
        
        tmp = combinations(sg_without_mtr,i+1, most_tied_region)
        tmp=list(tmp)
        sg_combi_redux+=tmp
        
        for k,j in enumerate( tmp):
            tmp_dict=[]
            for l,m in enumerate(j):
                tmp_dict+=regions_to_evaluate_dict[m]
            
            tmp[k]=list(np.unique(tmp_dict))
        
        sg_combi+=list(tmp)
        
    
    
    return (sg_combi,sg_combi_redux, most_tied_region, sg_without_mtr)


  
def flatten(l): 
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def generate_patches_from_multiple_regions(regions,
                                           regions_combination,
                                           rag,
                                           box,
                                           label_image,
                                           im_grey):
    """
    Se generan los parches con nucleos celulares
    """
    import matplotlib.pyplot as plt
    input_shape = im_grey.shape
    region_image = label_image.copy()
    
    
    labels = [i.label for i in regions]
    l=100001
    centre = regions[labels.index(regions_combination[0][-1])].centroid
    for n,combi in enumerate(regions_combination):
        
        region_image[:,:] = 0
        for k in combi:
            region_image[label_image == k] = 1
        region_image[region_image != 0] = im_grey[region_image != 0]
       
        bbox = [(int(centre[0]-box/2)),(int(centre[0]+box/2)),(int(centre[1]-box/2)),(int(centre[1]+box/2))]
        for i,j in enumerate(bbox):
            if j<0:
                bbox[i+1]=box
                bbox[i]=0
            if j>input_shape[0] and i==1:
                bbox[i-1]=input_shape[0]-box
                bbox[i]=input_shape[0]
            if j>input_shape[1] and i==3:
                bbox[i-1]=input_shape[1]-box
                bbox[i]=input_shape[1]
                    
                    
            
        sel_cel = region_image[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        sel_cel=resize(sel_cel, (192,192), mode="symmetric")
        sel_cel = sel_cel*255/sel_cel.max()
            
           
        col_sel_cel=np.matrix(sel_cel, "uint8")
        img=Image.fromarray(col_sel_cel)
        img.save("".join(["/tmp/SegNu/Samples/",str(l),".tif"]))
        l=l+1
        plt.close("all")
    return 

def get_optimized_cells(rag, regions_to_evaluate, unicell_regions,patch,label_image, image ):
    subgraphs=None
    sub_graphs=get_subgraphs(rag, regions_to_evaluate, subgraphs=[])
    
    num_regions_to_evaluate = len(sub_graphs)
    
    results = []
    for i in range(num_regions_to_evaluate):
        
        
        tmp_subgraph=sub_graphs[i]
        sub_graphs_dic={ j:[j] for j in tmp_subgraph}
        
        
        while len(tmp_subgraph)>0:
            regions_combination,sg_combi_redux, region_evaluated, sg_without_mtr=get_regions_combination(rag, sub_graphs_dic, tmp_subgraph)
            generate_patches_from_multiple_regions(unicell_regions,regions_combination,rag, patch, label_image,image)
            
            predictions = CNN_inference(192,len(regions_combination), argmax=False)
            
            tmp_index=np.argmax(predictions)
            sub_graphs_dic[region_evaluated]+=sg_combi_redux[tmp_index]
            sg_combi_redux[tmp_index]=list(np.unique(sg_combi_redux[tmp_index]))
            for m in sg_combi_redux[tmp_index]:
                sub_graphs_dic[region_evaluated]+=sub_graphs_dic[m]
                if m!=region_evaluated:
                    tmp=rag.merge_nodes(m,region_evaluated, in_place=True)
                    
                        
                    del sub_graphs_dic[m]
                    
                if m in tmp_subgraph:
                    del tmp_subgraph[tmp_subgraph.index(m)]
            
            for n in sg_without_mtr:
                if n not in sg_combi_redux[tmp_index]:
                    if len(rag[n])==1 and n in tmp_subgraph:
                        del tmp_subgraph[tmp_subgraph.index(n)]
            sub_graphs_dic[region_evaluated]=list(np.unique(sub_graphs_dic[region_evaluated]))
            rm_tmp_files()
            
            
            
            
        results+=sub_graphs_dic.values()
    return (results)

def save_instances_image(segmented_image, output_dir,file):
    """
    Saving the segmented and labeled image
 
    :param segmented_image: A numpy matrix. Labeled image.
	:param output_dir: String. Output directory.
	:param file: String. Input image path.
    """
    segmented_image=np.matrix(segmented_image, "uint8")
    img=Image.fromarray(segmented_image)
    img.save("".join([output_dir,"/Instances_images/",file.split(".")[0],".tif"]))

def convert_to_text(segmented_image):
    """
    """
    txt=""
    n=0
    de_cell=0
    init=0
    a,b=segmented_image.shape
    for j in range(b):
        for i in range(a):
            tmp=segmented_image[i,j]
            if tmp!=0:
                if de_cell==0:
                    de_cell=1
                    init=j*b+i+1
                n+=1	    
                if i>=a-1:
                    fin=n
                    de_cell=0
                    txt+=str(init)+" "+str(fin)
                    n=0
                else:
                    if segmented_image[i+1,j]!=tmp:
                        fin=n
                        de_cell=0
                        txt=txt+ " "+str(init)+" "+str(fin)
                        n=0
    		
    return(txt[1:])
    
def label_final_cell_proposed(final_proposes, label_image):
    if final_proposes==[]:
        return (label_image)
    else:
        tmp_prop = final_proposes[0]
        if len(tmp_prop)>1:
            label = tmp_prop[0]
            for j in tmp_prop[1:]:
                label_image[label_image==j]=label
                    
        return (label_final_cell_proposed(final_proposes[1:], label_image))
      

def patch_tune(images_list, PATCH_RATIO, tmp_regions=[]):
    """
    Tune the patch length
 
    :param images_list: A numpy matrix. Labeled image.
    :param PATCH_RATIO: Constant. Ratio between patch area against cell mean cell area.
    """
    if len(tmp_regions)>100 or images_list==[]:
        areas = [region.area for region in tmp_regions]
        mean = np.mean(areas)
        patch = int((PATCH_RATIO*mean)**(0.5))
        radio = int((mean/np.pi)**0.5)
        return (patch, radio)
    else:
        image = io.imread(images_list[0])
        if len(image.shape) == 3:
            image = image[:,:,2]                
        patch,regions_above_noise, label_image, threshold = initial_segmentation(image,PATCH_RATIO)
        generate_patches(regions_above_noise, patch,label_image,image)
        
        predictions = CNN_inference(192,len(regions_above_noise))
        
        rm_tmp_files()
        regions_without_debris = [j for i,j in enumerate(regions_above_noise) if  predictions[i]>0]
        tmp_regions += regions_without_debris
        return (patch_tune(images_list[1:], PATCH_RATIO, tmp_regions))


def run(PATCH_RATIO):
    wd, od, save_images = set_arg(sys.argv[1:])
    create_folders(od)
    #rm_tmp_files()
    extensions = ['.jpg', '.jpeg', '.tif', '.TIF', '.JPG', '.JPEG', ".png"]
    images_list = []
    with open(od+"/Results.txt", "w") as f:
        f.write("\t".join(["Image", "Number of cell nuclei"])+"\n")
    print("Loading files...\n")
    for file in os.listdir(wd):
        for extension in extensions:
            if file.endswith(extension):   
                images_list.append(os.path.join(wd, file))
    
    print("Tuning zoom patch...\n")        
    patch,radio = patch_tune(images_list, PATCH_RATIO, tmp_regions=[])
    print("Set new cell radio:"+str(radio))
    print("End tuning zoom patch.\n")
    t0 = time.time()
    for path_image in images_list:
        print ("Analyzing image: "+path_image.split("/")[-1])
        image = io.imread(path_image)
        if len(image.shape)==3:
            image = image[:,:,2]
        _,regions_above_noise, label_image, threshold = initial_segmentation(image,PATCH_RATIO)
        generate_patches(regions_above_noise, patch,label_image,image)
        predictions = CNN_inference(192,len(regions_above_noise))
        print(predictions)
        regions_without_debris = [j for i,j in enumerate(regions_above_noise) if  predictions[i]>0]
        predictions=[i for i in predictions if i!=0]
        cluster_regions = [j for i,j in enumerate(regions_without_debris) if  predictions[i]==2 ]
        unicell_regions = [i for i in regions_without_debris if i not in cluster_regions]
        rm_tmp_files()
        intances_image, final_regions = run_watershed_seg(radio,unicell_regions,cluster_regions,label_image, image, patch, threshold)
        del(unicell_regions,cluster_regions,label_image,threshold, predictions, regions_without_debris,regions_above_noise)
        save_instances_image(intances_image, od, path_image.split("/")[-1])
        with open(od+"/Results.txt", "a") as g:
            g.write("\t".join([path_image.split("/")[-1], str(len(final_regions))])+"\n")
        print("Number of cells found: "+str(len(final_regions)))

        if save_images:
            save_mod_images(image,intances_image, final_regions,od, path_image.split("/")[-1])

        del(image,intances_image, final_regions)
        
    t1 = time.time()
    print((t1-t0)/float(len(images_list)))

    

