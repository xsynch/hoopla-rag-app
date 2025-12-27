import argparse


from lib.describe_image import get_image_results

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    
    # image_parser = parser.add_parser(
    #     "--image", help="Perform RAG (search + generate answer) on an image"
    # )
    parser.add_argument("--image", type=str, help="Image path and name to use for search")

    # query_parser = parser.add_parser("--query",help="Query to use to search")
    parser.add_argument("--query",type=str,help="Actual Query to use for searching")
    


    args = parser.parse_args()
    
    run_multimodal_search(args.image,args.query)

    # match args.command:
    #     case "rag":            
    #         run_multimodal_search(args.image_path,args.query)

    #     case _:
    #         parser.print_help()

def run_multimodal_search(image,query):
    get_image_results(image,query)

if __name__ == "__main__":
    main()