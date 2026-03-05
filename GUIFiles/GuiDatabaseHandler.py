
import mysql.connector
import config


class GuiDatabaseHandler:

    def __init__(self):
        self.db = None
        self.cursor = None

    def start_database(self):
        self.db = mysql.connector.connect(
            host=config.HOST,
            user=config.USER,
            password=config.PASSWORD,
            database=config.DATABASE
        )
        self.cursor = self.db.cursor()

    def get_desc_file(self, video_path, language):
        self.start_database()
        if video_path:
            video_path = video_path.replace("/", "\\")
            db_command = ("SELECT df.path "
                          "FROM video_handler.des_files df "
                          "JOIN video_handler.video v ON df.video_id = v.id "
                          "WHERE v.path = %s and df.language = %s ")
            self.cursor.execute(db_command, (video_path, language))
            result = self.cursor.fetchone()
            if result is None:
                self.close_db()
                return None
            else:
                result = result[0]
                self.close_db()
                return result


    def get_translations(self):
        self.start_database()
        command = "SELECT sk_translation, en_translation, wid_name, parent from video_handler.translations"
        self.cursor.execute(command)
        translations = {}
        translation_records = self.cursor.fetchall()
        if translation_records:
            for sk_translation, en_translation, wid_name, parent in translation_records:
                translations[wid_name] = {
                    "sk_translation": sk_translation,
                    "en_translation": en_translation,
                    "parent": parent
                }
            self.close_db()
            return translations
        else:
            self.close_db()
            return None


    def check_video_file_exists(self, video_path):
        self.start_database()
        video_path = video_path.replace("/", "\\")
        command = "SELECT * FROM video_handler.video_files WHERE path = %s"
        self.cursor.execute(command, (video_path,))
        video_files = self.cursor.fetchall()
        if video_files:
            self.close_db()
            return True
        else:
            self.close_db()
            return False

    def insert_video_file(self, video_path):
        self.start_database()
        video_path = video_path.replace("/", "\\")
        command = "INSERT INTO video_handler.video_files (video_name, path) VALUES (%s, %s)"
        try:
            self.cursor.execute(command, (video_path, video_path))
            self.db.commit()
            self.close_db()
            return True
        except Exception as e:
            self.close_db()
            print("Error while inserting video_file")
            return False

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.db:
            self.db.close()
