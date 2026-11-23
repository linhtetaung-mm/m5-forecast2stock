# Deploy Stockdesk

Choose **Cloudflare Pages Direct Upload** for the course submission if the demo's data can be shared with anyone who has the link. It gives the grader a hosted HTTPS URL without a running AWS Lab or Python service. Use **AWS Academy Learner Lab with EC2** if the goal is to demonstrate the AWS setup during the presentation. The observed course lab shows a $50 budget, lists EC2, and restricts most services to us-east-1 and us-west-2. Its service list did not include Amplify. [Course Learner Lab](https://awsacademy.instructure.com/courses/196398/modules/items/19443645) · [AWS Learner Lab guide](https://d1.awsstatic.com/AWS%20Academy%20Learner%20Lab%20Educator%20Guide.pdf) · [Cloudflare Pages Free limits](https://developers.cloudflare.com/pages/platform/limits/)

## Prepare the website package

Stockdesk is the static replay served from `demo/dist`. It uses HTML, CSS and JavaScript, with a frozen study data file. The Python inventory scorer runs independently as a batch job and writes a recommendation CSV. Use the [batch and Docker guide](BATCH_DEPLOYMENT.md) to reproduce that part.

`study-data.js` includes item-level historical M5 sales for three teaching items and aggregate simulated results. Browser assets can be downloaded by whoever can access the site. Confirm that the dataset and course allow sharing these examples with the intended audience before uploading. If public sharing is unclear, use an authorized restricted course demonstration or prepare a synthetic-only public export. The website package excludes the repository, raw M5 CSVs, trained models, notebooks, order outputs and credentials. [M5 competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)

From Terminal on the Mac, run:

```sh
cd /Users/krown/m5-forecast2stock
node demo/test.mjs
package_dir=$(mktemp -d /private/tmp/stockdesk-release.XXXXXXXX)
zip -j "$package_dir/stockdesk.zip" \
  demo/dist/index.html demo/dist/style3.css demo/dist/app.js \
  demo/dist/replay.js demo/dist/study-data.js demo/dist/icon.svg
unzip -l "$package_dir/stockdesk.zip"
printf '%s\n' "$package_dir/stockdesk.zip"
```

The listing should contain six files with `index.html` at the archive root. Save or keep the ZIP until the submission is complete. The existing local demo is available at [localhost port 8770](http://127.0.0.1:8770/) when the local server is running.

## Option 1 · Free hosting on Cloudflare Pages

1. Open the Cloudflare dashboard. In **Workers & Pages**, choose **Create application** and find **Pages → Drag and drop your files** (Direct Upload).
2. Name the project, for example `stockdesk-ca1`. Upload the ZIP from the prepare step. Choose **Deploy site** or **Save and Deploy**, as shown.
3. Wait for a successful deployment and copy the HTTPS `pages.dev` URL. Use this URL in the course submission after checking the permitted audience. Test it in a private browser window.
4. To publish a correction, rebuild the ZIP and choose **Create a new deployment** under the existing project. Choose **Production** for the submission link.

Direct Upload needs no Git repository or build command. Cloudflare's dashboard accepts ZIP files, with up to 1,000 files and a 25 MiB limit per asset. A Direct Upload project cannot later switch to Git integration in place. [Cloudflare Pages Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/)

## Option 2 · AWS Academy Learner Lab with EC2

1. In the course lab, choose **Start Lab**, then **AWS** once ready. Use **US East (N. Virginia), us-east-1** so the lab's existing `vockey` is available.
2. In **EC2 → Launch instances**, create one dedicated instance with **Amazon Linux 2023** and an allowed micro instance such as `t3.micro` where offered. Select `vockey` and enable a public IPv4 address in a public subnet. Use a dedicated security group with **SSH (22)** from **My IP**. Open **HTTP (80)** only to the approved audience's IP or network range. The optional all-IPv4 HTTP rule would make the demo public. Use it only if publication is allowed. [AWS EC2 launch tutorial](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/tutorial-launch-a-test-ec2-instance.html) · [AWS security-group rules](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules-reference.html)
3. From the Learner Lab's **AWS Details**, download the PEM key. Keep it local. Copy the instance's current public IPv4 address from the console. In the **same Mac Terminal** where you created the ZIP, replace the address placeholder and run:

   ```sh
   ec2_host='REPLACE_WITH_EC2_PUBLIC_IPV4'
   chmod 400 "$HOME/Downloads/labsuser.pem"
   scp -i "$HOME/Downloads/labsuser.pem" "$package_dir/stockdesk.zip" \
     "ec2-user@$ec2_host:/tmp/stockdesk.zip"
   ssh -i "$HOME/Downloads/labsuser.pem" "ec2-user@$ec2_host"
   ```

4. In the SSH session on EC2, install the static web server and unpack the ZIP:

   ```sh
   sudo dnf install -y httpd unzip
   sudo unzip -o /tmp/stockdesk.zip -d /var/www/html
   sudo systemctl enable --now httpd
   curl -fI http://127.0.0.1/
   ```

   From an allowed network, open the EC2 public address over HTTP. These steps configure plain HTTP for a short demonstration. Use the HTTPS hosting route for a public link that will be graded later. [Apache on Amazon Linux 2023](https://docs.aws.amazon.com/linux/al2023/ug/ec2-lamp-amazon-linux-2023.html)

5. Check the lab timer before the presentation. A session ending can stop EC2, and the public IP can change after a restart. Use the current address at presentation time. Stop or terminate only the dedicated demo instance once it is no longer needed. [Course Learner Lab](https://awsacademy.instructure.com/courses/196398/modules/items/19443645)

## Verify the deployed link

- Open the root URL. The Order desk should show **Friday, Apr 1** and the initial suggestion of **7 units**.
- Run one review day and choose **Next day**. The review date should read **Saturday, Apr 2**. Confirm Manual mode accepts a whole-unit order.
- Open **Study results**, change the lead time and demand regime, and confirm all six named methods are visible in the legend and policy table.
- Check the active teal `style3.css` theme and test the link from a separate browser session or the grader's intended network. Stock, lost sales and costs are simulated.
