import hashlib
from supabase import create_client
from components.config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_hash(bytes):
    hasher = hashlib.sha256()
    hasher.update(bytes)
    return hasher.hexdigest()

def upload_misclassified_image(image, true_class, mime_type):
    image_bytes = image.read()
    new_hash = get_hash(image_bytes)
    new_filename = f"{new_hash}.jpg"
    path = f"Tmisclassified-images/{true_class}/{new_filename}"

    folders = supabase.storage.from_("misclassified-images").list("Tmisclassified-images")

    #Check for duplicate across all folders
    for folder in folders:
        folder_name = folder['name'].strip("/")
        files = supabase.storage.from_("misclassified-images").list(f"Tmisclassified-images/{folder_name}/")
        for file in files:
            if file["name"] == new_filename:
                st.warning("Image already uploaded")
                return

    supabase.storage.from_("misclassified-images").upload(path, image_bytes, {"content-type": mime_type})