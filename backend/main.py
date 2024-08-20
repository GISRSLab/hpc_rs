import logging.handlers
from typing import Annotated,Literal
from fastapi import FastAPI, Request, Response, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from starlette.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import algo.mytools.cv as mycv
import uvicorn
from uvicorn.server import logger
import algo.mytools.noise as mynoise
import os
import algo.mytools.cuda as mycuda
from datetime import datetime
import logging
import algo.filter.junzhi as junzhi
import algo.filter.ditong as ditong
import algo.mytools.read as myread
import algo.mytools.save as mysave


upload_dir = "./uploads/"
image_dir = "./static/images/"
# 创建文件夹用来存储上传文件
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def accord_images_periodic():
    logger.info(f'--删除多余图片，保持文件&图片一致-- {datetime.now()}')
    #Scan ./uploads
    uploads_file_list = os.listdir(upload_dir)
    jpg_file_list = map(pathFormat, uploads_file_list, ['jpg' for _ in range(len(uploads_file_list))])

    real_jpg_flist = os.listdir(image_dir)

    for item in real_jpg_flist:
        if item not in jpg_file_list:
            os.remove(os.path.join(image_dir, item)) 
            logger.info(f'删除图片: {item}')
    logger.info('--完成本次一致性处理--')


app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=[""]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def app_start():
    trigger = IntervalTrigger(seconds=60, start_date=datetime.now())
    scheduler.add_job(accord_images_periodic, trigger)
    scheduler.start()

def msg(message: str | list[str]) -> Response:
    return {"msg": message}

def pathFormat(path:str, format:str)->str:
    """
    @param path:str eg. "xxx.tif"
    @param format:str eg. "jpg"
    """
    pl = path.split(".")
    pl[-1] = format
    
    return ".".join(pl)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info('Access to /')
    return templates.TemplateResponse(request=request, name="file_upload.html")


@app.post("/file_upload", summary="文件上传")
async def file_upload(
    file: UploadFile,
):
    try:
        print(f"filename: {file.filename}")
        fpath = f"./uploads/{file.filename}"
        file_content = file.file.read()
        if not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(file_content)
        return msg("success")

    except Exception as e:
        print(e)
        return msg("fail")


@app.get("/files_list", summary="文件列表")
async def get_files_list():
    return msg(os.listdir("./uploads/"))


@app.post("/file_remove", summary="删除文件")
async def remove_file(filename: Annotated[str, Form()]):
    try:
        os.remove(f"./uploads/{filename}")
        return msg("success")
    except Exception as e:
        print(e)
        return msg("fail")


@app.get("/filter")
async def gpu(filename: str, method: str):
    try:
        path = os.path.join(upload_dir, filename)
    except Exception as e:
        print(f"/filter error: path--{path}")
        return msg({"status": "fail"})
    match method:
        case "mean":
            result_name = pathFormat(filename, "_mean_filter.TIF")
            result_path = os.path.join(upload_dir, result_name)
            if os.path.exists(result_path):
                return await GenImage(result_name)
            raster = myread.open_raster(path)
            result_raster = junzhi.main(raster)
            mysave.write_raster(result_path, result_raster, path)
            result = await GenImage(result_name)
            return result
        case "lower":
            result_name = pathFormat(filename, "_lower_filter.TIF")
            result_path = os.path.join(upload_dir, result_name)
            if os.path.exists(result_path):
                return await GenImage(result_name)
            raster = myread.open_raster(path)
            result_raster = ditong.main(raster)
            mysave.write_raster(result_path, result_raster, path)
            result = await GenImage(result_name)
            return result


@app.get("/genimage", summary="生成jpg")
async def GenImage(filename: str):
    file_path = os.path.join(upload_dir, filename)

    filename_list = filename.split(".")
    filename_list[-1] = "jpg"
    file_jpg = ".".join(filename_list)
    out_path = os.path.join(image_dir, file_jpg)
    if os.path.exists(out_path):
        return msg({"status": "success", "output": out_path.split("/")[-1]})
    if os.path.exists(file_path):
        result = mycv.stretchImg(file_path, out_path)
        if result:
            return msg({"status": "success", "output": out_path.split("/")[-1]})
        else:
            return msg({"status": "fail"})
    else:
        print(file_path)
        return msg({"status": "File not found"})

class Noise(BaseModel):
    filename: str
    std:Literal[200, 400, 600, 800]=400

@app.post("/noise", summary="加噪音")
async def AddNoise(noise:Noise):
    file_path = os.path.join(upload_dir, noise.filename)

    filename_list = noise.filename.split(".")
    filename_list[-2] += "_noise_" + str(noise.std)
    out_name = ".".join(filename_list)
    out_path = os.path.join(upload_dir, out_name)
    status = await mynoise.add_noise_to_tif(file_path, out_path, 0, noise.std)
    if status:
        result = await GenImage(out_name)
        return result
         
@app.get("/download", summary="下载文件")
async def Download(filename: str):
    file_path = os.path.join(upload_dir, filename)

    return FileResponse(file_path, filename=filename)

    
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=5000, reload=True)