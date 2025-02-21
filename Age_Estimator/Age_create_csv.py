import csv
import os
import pandas as pd

# Create a csv file which contains labels
def create_csv(dataset_folder):
    image_files = os.listdir(dataset_folder)
    header = ['image_name', 'age', 'ethnicity', 'gender']
    with open('/home/bassant-raafat/task/AgeDetector_Facematching/Age_Estimator/csv_dataset_AgeDetect/utkface_dataset.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, image_file in enumerate(image_files):
            if len(image_file.split('_')) < 4:
                continue
            
            # convert values to int with map function
            age, gender, ethnicity = map(int, image_file.split('_')[:3])
    
            gender = 'Male' if gender == 0 else 'Female'
            if ethnicity != str:
                ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][ethnicity]

            if age < 85:
                data = [image_file, age, ethnicity, gender]
                writer.writerow(data)


# Replace with the actual path to your UTK dataset images folder
dataset_folder = '/home/bassant-raafat/task/AgeDetector_Facematching/Datasets/utkcropped'
create_csv(dataset_folder)

df = pd.read_csv('/home/bassant-raafat/task/AgeDetector_Facematching/Age_Estimator/csv_dataset_AgeDetect/utkface_dataset.csv')
print(f'Dataframe length: {len(df)}')
print()
print(df.head())