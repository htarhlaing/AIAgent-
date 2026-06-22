import os
import hashlib
from AI_agent.utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def get_file_md5_hex(filepath):
  if not os.path.exists(filepath):
    logger.error(f"文件不存在:{filepath}")
    return 

  if not os.path.isfile(filepath):
    logger.error(f"文件不是普通文件:{filepath}")
    return 

    # 计算文件的md5值
  md5_obj=hashlib.md5()
  chunk_size=4096
  try:
    with open(filepath,"rb") as f: 
      while chunk:=f.read(chunk_size):
        md5_obj.update(chunk)
      md5_hex=md5_obj.hexdigest()
      return md5_hex

      # while chunk:
      #   md5_obj.update(chunk)
      #chunk=f.read(chunk_size)
  except Exception as e:
    logger.error(f"计算文件md5值失败:{filepath}，错误信息:{e}")
    return 
  

def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):
  files=[]
  #判断是否是允许的文件类型
  if not os.path.exists(path):
    logger.error(f"[listdir_with_allowed_type]路径不存在:{path}")
    return allowed_types
  for f in os.listdir(path):
    if f.endswith(allowed_types):
      files.append(os.path.join(path,f))

  return tuple(files )


def pdf_loader(filepath:str,passwd=None)->list[Document]:
  return PyPDFLoader(filepath,passwd).load()
  

def txt_loader(filepath:str)->list[Document]:
  return TextLoader(filepath,encoding="utf-8").load()
