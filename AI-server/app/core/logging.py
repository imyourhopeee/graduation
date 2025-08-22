#서버 전체의 로깅 포맷/레벨을 설정하는 모듈.
# (INFO, DEBUG, ERROR 로그를 깔끔하게 출력하도록 통일)

import logging, sys

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
