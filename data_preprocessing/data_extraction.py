import os
import py7zr

def extract_7z_files(zip_files, extract_to_folder):
    os.makedirs(extract_to_folder, exist_ok=True)
    for zip_file in zip_files:
        print(f"Extracting {zip_file}...")
        with py7zr.SevenZipFile(zip_file, mode='r') as archive:
            archive.extractall(path=extract_to_folder)
    print("Extraction complete.")

if __name__ == "__main__":
    zip_files = [
        r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\Original_zip_files\EC press releases part 1.7z",
        r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\Original_zip_files\EC press releases part 2.7z",
        r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\Original_zip_files\EC press releases part 3.7z",
        r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\Original_zip_files\EC press releases part 4.7z"
    ]
    extract_to_folder = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\raw_data"
    extract_7z_files(zip_files, extract_to_folder)


    