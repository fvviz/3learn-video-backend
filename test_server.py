import asyncio
import aiohttp
import json

# Test image URLs
IMAGE_URLS = [
    "https://hotfr.s3.us-west-2.amazonaws.com/630d13a2-94ef-4755-bb1d-6ab754d23ec7_user_2tw5JcWFZxfWFGXzEWwLqh5FMfl_1741251371761_0.jpg",
    "https://hotfr.s3.us-west-2.amazonaws.com/630d13a2-94ef-4755-bb1d-6ab754d23ec7_user_2tw5JcWFZxfWFGXzEWwLqh5FMfl_1741251377767_1.jpg",
    "https://hotfr.s3.us-west-2.amazonaws.com/630d13a2-94ef-4755-bb1d-6ab754d23ec7_user_2tw5JcWFZxfWFGXzEWwLqh5FMfl_1741251382762_2.jpg",
    "https://hotfr.s3.us-west-2.amazonaws.com/630d13a2-94ef-4755-bb1d-6ab754d23ec7_user_2tw5JcWFZxfWFGXzEWwLqh5FMfl_1741251387762_3.jpg",
    "https://hotfr.s3.us-west-2.amazonaws.com/630d13a2-94ef-4755-bb1d-6ab754d23ec7_user_2tw5JcWFZxfWFGXzEWwLqh5FMfl_1741251392758_4.jpg"
]

BASE_URL = "http://localhost:8000"

async def send_analysis_request(session, job_id, batch_urls, batch_num):
    print(f"\nSending batch {batch_num} with {len(batch_urls)} images")
    async with session.post(
        f"{BASE_URL}/analyze_student_images",
        json={
            "job_id": job_id,
            "image_urls": batch_urls
        }
    ) as response:
        result = await response.json()
        print(f"Response for batch {batch_num}: {result}")
        return result

async def test_workflow():
    async with aiohttp.ClientSession() as session:
        # 1. Create a new job
        job_id = "test_job_1"
        print(f"\n1. Creating job: {job_id}")
        async with session.post(
            f"{BASE_URL}/create_job",
            json={"job_id": job_id}
        ) as response:
            result = await response.json()
            print(f"Response: {result}")

        # 2. Send multiple image analysis requests concurrently
        print("\n2. Sending multiple image analysis requests concurrently")
        tasks = []
        for i in range(3):  # Test with 3 batches of images
            batch_urls = IMAGE_URLS[i:i+2]  # Take 2 images at a time
            task = send_analysis_request(session, job_id, batch_urls, i+1)
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        print("\nAll requests sent:", results)

        # 3. Wait a bit for processing to complete
        print("\n3. Waiting for processing to complete...")
        await asyncio.sleep(10)

        # 4. Get job analysis
        print("\n4. Getting job analysis")
        async with session.post(
            f"{BASE_URL}/analyze_job",
            json={"job_id": job_id}
        ) as response:
            result = await response.json()
            print("\nFinal Analysis:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_workflow()) 