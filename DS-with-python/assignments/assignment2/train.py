import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from sklearn.feature_extraction import image
import numpy as np

parser = ArgumentParser()
parser.add_argument('train_data_file',type=str)
parser.add_argument('model_file',type=str)
parser.add_argument('n_pixels',type=int)
args = parser.parse_args()

#saving the models
filename = args.model_file

#prepare data
# Below, we are using the full dataset.
images, face_ids = joblib.load(f'{args.train_data_file}')
print(f'Images number and dimensions are {images.shape}\n')

print('Preparing data...\n')
#rotating images for 30,50 and 90 pixels size
for p in [30,50,90]:
    if p == 30:
        patch_extractor = image.PatchExtractor(patch_size=(p, p), max_patches=8, random_state=0)
        sub_images1 = patch_extractor.transform(images)
        j = 0
        sub_images30 = sub_images1.copy()
        y_train_30 = []
        for i in range(len(sub_images30)):
            if j == 0:
                sub_images30[i] = np.rot90(sub_images30[i], k=j)
                y_train_30.append(j)
                j += 1
            elif j == 1:
                sub_images30[i] = np.rot90(sub_images30[i], k=j)
                y_train_30.append(j)
                j += 1
            elif j == 2:
                sub_images30[i] = np.rot90(sub_images30[i], k=j)
                y_train_30.append(j)
                j += 1
            elif j == 3:
                sub_images30[i] = np.rot90(sub_images30[i], k=j)
                y_train_30.append(j)
                j = 0
    
    elif p == 50:
        patch_extractor = image.PatchExtractor(patch_size=(p, p), max_patches=7, random_state=0)
        sub_images2 = patch_extractor.transform(images)
        j = 0
        sub_images50 = sub_images2.copy()
        y_train_50 = []
        for i in range(len(sub_images50)):
            if j == 0:
                sub_images50[i] = np.rot90(sub_images50[i], k=j)
                y_train_50.append(j)
                j += 1
            elif j == 1:
                sub_images50[i] = np.rot90(sub_images50[i], k=j)
                y_train_50.append(j)
                j += 1
            elif j == 2:
                sub_images50[i] = np.rot90(sub_images50[i], k=j)
                y_train_50.append(j)
                j += 1
            elif j == 3:
                sub_images50[i] = np.rot90(sub_images50[i], k=j)
                y_train_50.append(j)
                j = 0
    
    elif p == 90:
        patch_extractor = image.PatchExtractor(patch_size=(p, p), max_patches=4, random_state=0)
        sub_images3 = patch_extractor.transform(images)
        j = 0
        sub_images90 = sub_images3.copy()
        y_train_90 = []
        for i in range(len(sub_images90)):
            if j == 0:
                sub_images90[i] = np.rot90(sub_images90[i], k=j)
                y_train_90.append(j)
                j += 1
            elif j == 1:
                sub_images90[i] = np.rot90(sub_images90[i], k=j)
                y_train_90.append(j)
                j += 1
            elif j == 2:
                sub_images90[i] = np.rot90(sub_images90[i], k=j)
                y_train_90.append(j)
                j += 1
            elif j == 3:
                sub_images90[i] = np.rot90(sub_images90[i], k=j)
                y_train_90.append(j)
                j = 0

#saving prepared data to dictionary train_data
train_data = {}
train_data[30] = { 'x_train':sub_images30,
                    'y_train': y_train_30}

train_data[50] = { 'x_train':sub_images50,
                    'y_train': y_train_50}

train_data[90] = { 'x_train':sub_images90,
                    'y_train': y_train_90}

print('Now, proceeding to training phase...\n')

if args.n_pixels == 30:
    #Training 30 pixel model
    best_params30 = {}
    acc1 = 0
    features_30 = train_data[30]['x_train'].reshape(len(train_data[30]['x_train']),-1)
    x_train,x_val,y_train,y_val = train_test_split(features_30,train_data[30]['y_train'],
                                                test_size=0.2,random_state=42)
    n_components = [10,20,30]
    hidden_layer_sizes = [(100,), (200,), (300,),(100,100)]
    print('Training 30 pixel model...')
    for i in n_components:
        for j in hidden_layer_sizes:
            pipeline = Pipeline([
                ('pca', PCA(n_components=i)),
                ('nn',MLPClassifier(hidden_layer_sizes=j,max_iter = 1000, random_state = 0, early_stopping=True))
            ])
            pipeline.fit(x_train,y_train)
            acc = pipeline.score(x_val,y_val)
            if acc > acc1:
                best_params30['n_components'] = i
                best_params30['hidden_layer_sizes'] = j
                acc1 = acc

    pipeline30 = Pipeline([
                ('pca', PCA(n_components=best_params30['n_components'])),
                ('nn',MLPClassifier(hidden_layer_sizes=best_params30['hidden_layer_sizes'],max_iter = 1000, random_state = 0, early_stopping=True))
            ])
    print(f'Best model for 30 pixels : {pipeline30}')
    pipeline30.fit(x_train,y_train)
    print(f'Best validation accuracy for 30 pixel model = {pipeline30.score(x_val,y_val)*100:0.2f} %\n')
    joblib.dump(pipeline30,f'.\\{filename}')
    print(f'30-pixel model saved to {filename}\n')

elif args.n_pixels == 50:
    #Training 50 pixel model
    best_params50 = {}
    acc1 = 0
    features_50 = train_data[50]['x_train'].reshape(len(train_data[50]['x_train']),-1)
    x_train,x_val,y_train,y_val = train_test_split(features_50,train_data[50]['y_train'],
                                                test_size=0.2,random_state=42)
    n_components = [35,45,50]
    hidden_layer_sizes = [(100,), (200,), (300,),(100,100)]
    print('Training 50 pixel model...')
    for i in n_components:
        for j in hidden_layer_sizes:
            pipeline = Pipeline([
                ('pca', PCA(n_components=i)),
                ('nn',MLPClassifier(hidden_layer_sizes=j,max_iter = 1000, random_state = 0, early_stopping=True))
            ])
            pipeline.fit(x_train,y_train)
            acc = pipeline.score(x_val,y_val)
            if acc > acc1:
                best_params50['n_components'] = i
                best_params50['hidden_layer_sizes'] = j
                acc1 = acc

    pipeline50 = Pipeline([
                ('pca', PCA(n_components=best_params50['n_components'])),
                ('nn',MLPClassifier(hidden_layer_sizes=best_params50['hidden_layer_sizes'],max_iter = 1000, random_state = 0, early_stopping=True))
            ])
    print(f'Best model for 50 pixels : {pipeline50}')
    pipeline50.fit(x_train,y_train)
    print(f'Best validation accuracy for 50 pixel model = {pipeline50.score(x_val,y_val)*100:0.2f} %\n')
    joblib.dump(pipeline50,f'.\\{filename}')
    print(f'50-pixel model saved to {filename}\n')

elif args.n_pixels == 90:   
    #Training 90 pixel model
    best_params90 = {}
    acc1 = 0
    features_90 = train_data[90]['x_train'].reshape(len(train_data[90]['x_train']),-1)
    x_train,x_val,y_train,y_val = train_test_split(features_90,train_data[90]['y_train'],
                                                test_size=0.2,random_state=42)
    n_components = [60,70,80]
    hidden_layer_sizes = [(100,), (200,), (300,),(100,100)]
    print('Training 90 pixel model...')
    for i in n_components:
        for j in hidden_layer_sizes:
            pipeline = Pipeline([
                ('pca', PCA(n_components=i)),
                ('nn',MLPClassifier(hidden_layer_sizes=j,max_iter = 1000, random_state = 0, early_stopping=True))
            ])
            pipeline.fit(x_train,y_train)
            acc = pipeline.score(x_val,y_val)
            if acc > acc1:
                best_params90['n_components'] = i
                best_params90['hidden_layer_sizes'] = j
                acc1 = acc

    pipeline90 = Pipeline([
                ('pca', PCA(n_components=best_params90['n_components'])),
                ('nn',MLPClassifier(hidden_layer_sizes=best_params90['hidden_layer_sizes'],max_iter = 1000, random_state = 0, early_stopping=True))
            ])
    print(f'Best model for 90 pixels : {pipeline90}')
    pipeline90.fit(x_train,y_train)
    print(f'Best validation accuracy for 90 pixel model = {pipeline90.score(x_val,y_val)*100:0.2f} %\n')
    joblib.dump(pipeline90,f'.\\{filename}')
    print(f'90-pixel model saved to {filename}')