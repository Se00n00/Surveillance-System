import sqlite3
import json
import numpy as np
import cv2

class ImageRecords:
    def __init__(self, db_path='images.db'):
        """
        Initialize the ImageRecords database.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """
        Create the images table if it doesn't exist.
        """
        query = """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL,
            metadata TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_image(self, image, metadata=None):
        """
        Add a single image to the database.

        Args:
            image (numpy.ndarray): Image to store.
            metadata (dict, optional): Metadata as a dictionary.
        """
        image_data = cv2.imencode('.png', image)[1].tobytes()
        metadata_json = json.dumps(metadata) if metadata else None

        query = "INSERT INTO images (image, metadata) VALUES (?, ?)"
        self.conn.execute(query, (image_data, metadata_json))
        self.conn.commit()

    def add_images(self, images, metadata_list=None):
        """
        Add multiple images to the database.

        Args:
            images (list of numpy.ndarray): List of images.
            metadata_list (list of dict, optional): List of metadata.
        """
        if metadata_list is None:
            metadata_list = [None] * len(images)

        for image, metadata in zip(images, metadata_list):
            self.add_image(image, metadata)

    def delete_image(self, image_id):
        """
        Delete an image by its ID.

        Args:
            image_id (int): ID of the image to delete.
        """
        query = "DELETE FROM images WHERE id = ?"
        self.conn.execute(query, (image_id,))
        self.conn.commit()

    def get_image(self, image_id):
        """
        Retrieve an image and its metadata by ID.

        Args:
            image_id (int): ID of the image.

        Returns:
            tuple: (image (numpy.ndarray), metadata (dict))
        """
        query = "SELECT image, metadata FROM images WHERE id = ?"
        cursor = self.conn.execute(query, (image_id,))
        result = cursor.fetchone()

        if result is None:
            return None, None

        image_data, metadata_json = result
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        metadata = json.loads(metadata_json) if metadata_json else None

        return image_array, metadata

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()
