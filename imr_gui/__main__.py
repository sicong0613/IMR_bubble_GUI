import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows for ProcessPoolExecutor
    from imr_gui.app import run
    run()

